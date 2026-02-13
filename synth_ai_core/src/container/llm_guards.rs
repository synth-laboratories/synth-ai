use url::Url;

const PROVIDER_DOMAINS: [&str; 7] = [
    "api.openai.com",
    "api.anthropic.com",
    "api.groq.com",
    "generativelanguage.googleapis.com",
    "api.cohere.ai",
    "api.together.xyz",
    "api.perplexity.ai",
];

pub fn is_direct_provider_call(url: &str) -> bool {
    let url = url.trim();
    if url.is_empty() {
        return false;
    }
    let parsed = Url::parse(url).ok();
    let host = match parsed.and_then(|u| u.host_str().map(|s| s.to_string())) {
        Some(host) => host,
        None => return false,
    };

    PROVIDER_DOMAINS
        .iter()
        .any(|domain| host == *domain || host.ends_with(&format!(".{domain}")))
}
