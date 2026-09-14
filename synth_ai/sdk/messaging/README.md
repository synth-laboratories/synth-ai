# Messaging

`SynthClient().messaging` and `AsyncSynthClient().messaging` expose typed threads,
history, enrollments and grants over the Synth backend using the existing API
credential. No MQ endpoints, signing keys or admin tokens are accepted.

Publication returns an accepted Message, not a delivered/executed receipt. Reuse
its idempotency key to retry the same logical message; altered input conflicts.
Thread create uses ensure semantics, not title updates. Device sign-in/credential
handling stays in Workshop. Owner history cannot substitute for device-grant
access tests. Methods do not automatically retry uncertain mutations.

Authority: backend `specifications/workshop-messaging-api.md` and its `/api/v1/mq`
routes. Tests live in the sibling testing repository.
