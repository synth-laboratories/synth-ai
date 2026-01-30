use thiserror::Error;

#[derive(Debug, Error)]
pub enum TunnelError {
    #[error("tunnel config error: {0}")]
    Config(String),
    #[error("tunnel API error: {0}")]
    Api(String),
    #[error("lease error: {0}")]
    Lease(String),
    #[error("connector error: {0}")]
    Connector(String),
    #[error("gateway error: {0}")]
    Gateway(String),
    #[error("local app error: {0}")]
    LocalApp(String),
    #[error("dns error: {0}")]
    Dns(String),
    #[error("process error: {0}")]
    Process(String),
    #[error("websocket error: {0}")]
    WebSocket(String),
}

impl TunnelError {
    pub fn config(msg: impl Into<String>) -> Self {
        TunnelError::Config(msg.into())
    }
    pub fn api(msg: impl Into<String>) -> Self {
        TunnelError::Api(msg.into())
    }
    pub fn lease(msg: impl Into<String>) -> Self {
        TunnelError::Lease(msg.into())
    }
    pub fn connector(msg: impl Into<String>) -> Self {
        TunnelError::Connector(msg.into())
    }
    pub fn gateway(msg: impl Into<String>) -> Self {
        TunnelError::Gateway(msg.into())
    }
    pub fn local(msg: impl Into<String>) -> Self {
        TunnelError::LocalApp(msg.into())
    }
    pub fn dns(msg: impl Into<String>) -> Self {
        TunnelError::Dns(msg.into())
    }
    pub fn process(msg: impl Into<String>) -> Self {
        TunnelError::Process(msg.into())
    }
    pub fn websocket(msg: impl Into<String>) -> Self {
        TunnelError::WebSocket(msg.into())
    }
}
