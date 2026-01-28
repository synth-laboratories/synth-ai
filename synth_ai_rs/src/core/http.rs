use std::sync::OnceLock;
use std::time::Duration;

static SHARED_HTTP_CLIENT: OnceLock<reqwest::Client> = OnceLock::new();

pub fn shared_client() -> &'static reqwest::Client {
    SHARED_HTTP_CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .pool_max_idle_per_host(20)
            .pool_idle_timeout(Duration::from_secs(30))
            .build()
            .expect("failed to build reqwest client")
    })
}
