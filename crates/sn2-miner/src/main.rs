#[cfg(target_os = "linux")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// Mimalloc reads option env vars on first option access (lazy). The default
// `purge_delay` of ~1s churns the page tables under our proving workload's
// allocation frequency (witness/proof buffers allocated and freed across
// many concurrent blocking threads), the same single-thread allocator
// bottleneck the validator observed on its dispatch workload (38k
// mmap/munmap syscalls per 3s on mainnet before this fix). Setting the env
// var from a constructor that runs before main() (and crucially before the
// tokio runtime build) captures the desired cadence before any sustained
// allocation. Operators can still override by setting the env var
// explicitly in their process environment.
#[cfg(target_os = "linux")]
#[ctor::ctor]
fn configure_mimalloc_purge_delay() {
    // SAFETY: ctor runs single-threaded before main; no other thread can
    // race on environment state at this point.
    unsafe {
        if std::env::var_os("MIMALLOC_PURGE_DELAY").is_none() {
            std::env::set_var("MIMALLOC_PURGE_DELAY", "60000");
        }
    }
}

mod cli;
mod dsperse;
mod handlers;
mod lightning_server;
mod wai_known_constants;

use std::net::IpAddr;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;
use tokio::signal::unix::{signal, SignalKind};
use tokio::sync::watch;
use tracing::{error, info, warn};

use crate::cli::Cli;

#[tokio::main]
async fn main() -> Result<()> {
    rustls::crypto::ring::default_provider()
        .install_default()
        .expect("failed to install rustls CryptoProvider");

    let cli = Cli::parse();

    sn2_types::init_tracing(&cli.log_level);

    info!(version = sn2_types::SOFTWARE_VERSION, "sn2-miner");

    if cli.loopback {
        return run_loopback(cli).await;
    }

    let (shutdown_tx, mut shutdown_rx) = watch::channel(false);

    if !cli.no_auto_update && option_env!("SN2_RELEASE_CHANNEL") == Some("mainnet") {
        let _update_handle =
            sn2_chain::auto_update::spawn_update_loop("sn2-miner", shutdown_tx.clone());
    }

    info!(
        netuid = cli.netuid,
        network = %cli.network,
        "starting sn2-miner"
    );

    let wallet = std::sync::Arc::new(
        sn2_chain::Wallet::from_paths(
            &cli.wallet_name,
            &cli.wallet_hotkey,
            cli.wallet_path.as_deref(),
        )
        .context("loading wallet")?,
    );

    let endpoint =
        sn2_chain::resolve_endpoint(&cli.network, cli.subtensor_chain_endpoint.as_deref());

    let chain_client = sn2_chain::connect_chain(&endpoint).await?;

    let registration = sn2_chain::Registration::new(cli.netuid);

    let mut metagraph = sn2_chain::Metagraph::new(cli.netuid);
    metagraph
        .sync(&chain_client)
        .await
        .context("initial metagraph sync")?;
    // Axon registration opens its own connections (see `register_axon`).
    drop(chain_client);

    anyhow::ensure!(
        metagraph.get_uid_by_hotkey(wallet.hotkey_ss58()).is_some(),
        "hotkey {} is not registered on subnet {}. Register with: btcli subnets register --netuid {} --network {}",
        wallet.hotkey_ss58(),
        cli.netuid,
        cli.netuid,
        cli.network,
    );

    let external_ip = match resolve_external_ip(cli.external_ip.as_deref()).await {
        Ok(ip) => Some(ip),
        Err(e) if cli.external_ip.is_none() => {
            warn!(
                error = ?e,
                "external IP autodetection failed; skipping serve_axon for this boot"
            );
            None
        }
        Err(e) => return Err(e),
    };

    let quic_port = cli.axon_port;
    anyhow::ensure!(quic_port != 0, "QUIC port must be non-zero");

    let dsperse = dsperse::DSperseClient::new(cli.circuit_cache_dir.as_deref());

    let circuit_store = init_circuit_store(
        false,
        &cli.additional_circuits,
        cli.circuit_cache_dir.as_deref(),
    )
    .await;

    let handlers = handlers::MinerHandlers::new(dsperse, circuit_store);
    let handlers = std::sync::Arc::new(handlers);

    let handler_timeout = cli.handler_timeout;
    let quic_handle = {
        let handlers = handlers.clone();
        let hotkey = wallet.hotkey_ss58().to_string();
        let w_name = wallet.name.clone();
        let w_path = wallet.wallet_path.clone();
        let w_hotkey = wallet.hotkey_name.clone();
        tokio::spawn(async move {
            lightning_server::run_lightning_server(
                &hotkey,
                &w_name,
                &w_path,
                &w_hotkey,
                "0.0.0.0",
                quic_port,
                handler_timeout,
                handlers,
                true,
            )
            .await
        })
    };

    // Registering an axon waits for block finalization and may retry for
    // minutes, so it runs off the path that watches the server and signals.
    if let Some(external_ip) = external_ip {
        tokio::spawn(register_axon(
            registration,
            endpoint.clone(),
            wallet.clone(),
            external_ip,
            quic_port,
        ));
    }

    info!(
        hotkey = %wallet.hotkey_ss58(),
        quic_port = quic_port,
        "miner running"
    );

    let mut sigterm = signal(SignalKind::terminate()).context("registering SIGTERM handler")?;

    tokio::select! {
        r = quic_handle => {
            r?.context("QUIC server")?;
        }
        _ = tokio::signal::ctrl_c() => {
            info!("shutting down miner");
        }
        _ = sigterm.recv() => {
            info!("received SIGTERM, shutting down miner");
        }
        _ = async { loop { shutdown_rx.changed().await.ok()?; if *shutdown_rx.borrow() { return Some(()); } } } => {
            info!("shutting down miner for auto-update restart");
        }
    }

    Ok(())
}

const SERVE_AXON_TIMEOUT: Duration = Duration::from_secs(120);
const SERVE_AXON_ATTEMPTS: u32 = 8;
const SERVE_AXON_FIRST_BACKOFF: Duration = Duration::from_secs(15);
const SERVE_AXON_MAX_BACKOFF: Duration = Duration::from_secs(300);

/// Publishes the miner's axon, retrying with backoff.
///
/// Every attempt opens its own chain connection. The startup connection sits
/// idle while a cold circuit cache downloads, which can take many minutes, and
/// the chain RPC client neither pings nor reconnects: once the endpoint or a
/// NAT on the way drops that idle socket, every call on it fails for good. A
/// fresh connection per attempt avoids depending on it. Retrying after a
/// timeout is safe because `serve_axon` first reads the chain and skips the
/// extrinsic when the axon is already registered with this IP and port.
async fn register_axon(
    registration: sn2_chain::Registration,
    endpoint: String,
    wallet: std::sync::Arc<sn2_chain::Wallet>,
    external_ip: IpAddr,
    port: u16,
) {
    let mut backoff = SERVE_AXON_FIRST_BACKOFF;
    for attempt in 1..=SERVE_AXON_ATTEMPTS {
        let served = tokio::time::timeout(SERVE_AXON_TIMEOUT, async {
            let client = sn2_chain::connect_chain(&endpoint).await?;
            registration
                .serve_axon(&client, &wallet, external_ip, port, 4)
                .await
        })
        .await;
        let error = match served {
            Ok(Ok(())) => return,
            // `{:#}` prints the whole context chain; the outermost context
            // alone ("fetching latest block") hides the cause.
            Ok(Err(e)) => format!("{e:#}"),
            Err(_) => format!("timed out after {}s", SERVE_AXON_TIMEOUT.as_secs()),
        };
        if attempt == SERVE_AXON_ATTEMPTS {
            error!(
                hotkey = %wallet.hotkey_ss58(),
                port,
                attempts = attempt,
                error,
                "serve_axon failed; giving up, validators cannot find this miner until it is restarted"
            );
            return;
        }
        warn!(
            hotkey = %wallet.hotkey_ss58(),
            port,
            attempt,
            retry_in_secs = backoff.as_secs(),
            error,
            "serve_axon failed; retrying with a new chain connection"
        );
        tokio::time::sleep(backoff).await;
        backoff = (backoff * 2).min(SERVE_AXON_MAX_BACKOFF);
    }
}

async fn run_loopback(cli: Cli) -> Result<()> {
    info!(
        port = cli.axon_port,
        "starting miner in loopback mode (no chain interaction)"
    );

    let wallet = sn2_chain::Wallet::from_paths(
        &cli.wallet_name,
        &cli.wallet_hotkey,
        cli.wallet_path.as_deref(),
    )
    .context("loading wallet")?;

    let dsperse = dsperse::DSperseClient::new(cli.circuit_cache_dir.as_deref());

    let circuit_store = init_circuit_store(
        true,
        &cli.additional_circuits,
        cli.circuit_cache_dir.as_deref(),
    )
    .await;

    let handlers = handlers::MinerHandlers::new(dsperse, circuit_store);
    let handlers = std::sync::Arc::new(handlers);

    let handler_timeout = cli.handler_timeout;
    let quic_handle = {
        let handlers = handlers.clone();
        let host = cli.axon_host.clone();
        let port = cli.axon_port;
        let hotkey_ss58 = wallet.hotkey_ss58().to_string();
        let w_name = wallet.name.clone();
        let w_path = wallet.wallet_path.clone();
        let w_hotkey = wallet.hotkey_name.clone();
        tokio::spawn(async move {
            lightning_server::run_lightning_server(
                &hotkey_ss58,
                &w_name,
                &w_path,
                &w_hotkey,
                &host,
                port,
                handler_timeout,
                handlers,
                false,
            )
            .await
        })
    };

    info!(port = cli.axon_port, "miner loopback running");

    let mut sigterm = signal(SignalKind::terminate()).context("registering SIGTERM handler")?;

    tokio::select! {
        r = quic_handle => {
            r?.context("QUIC server")?;
        }
        _ = tokio::signal::ctrl_c() => {
            info!("shutting down miner");
        }
        _ = sigterm.recv() => {
            info!("received SIGTERM, shutting down miner");
        }
    }

    Ok(())
}

async fn resolve_external_ip(override_ip: Option<&str>) -> Result<IpAddr> {
    if let Some(ip) = override_ip {
        let parsed: IpAddr = ip.parse().context("parsing --external-ip")?;
        return require_ipv4(parsed);
    }
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .context("building HTTP client for external-IP detection")?;
    let resp = client
        .get("https://api4.ipify.org")
        .send()
        .await
        .context("detecting external IP via api4.ipify.org")?
        .text()
        .await
        .context("reading external IP response body")?;
    let parsed: IpAddr = resp
        .trim()
        .parse()
        .with_context(|| format!("parsing detected IP: {resp}"))?;
    require_ipv4(parsed)
}

fn require_ipv4(ip: IpAddr) -> Result<IpAddr> {
    match ip {
        IpAddr::V4(_) => Ok(ip),
        IpAddr::V6(_) => {
            anyhow::bail!(
                "external IP must be IPv4 (axon registration does not support IPv6): {ip}"
            )
        }
    }
}

async fn init_circuit_store(
    loopback: bool,
    additional_circuits: &[String],
    cache_dir_override: Option<&str>,
) -> sn2_circuit_store::CircuitStore {
    let mut store = sn2_circuit_store::CircuitStore::new(
        None,
        loopback,
        additional_circuits.to_vec(),
        cache_dir_override,
    );
    if let Err(e) = store.load_circuits().await {
        warn!(error = %e, "failed to load circuits from cache");
    }
    for id in additional_circuits {
        if let Err(e) = store.ensure_circuit(id).await {
            warn!(id = %id, error = %e, "failed to preload pinned circuit");
        }
    }
    store
}
