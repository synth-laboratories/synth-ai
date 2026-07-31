"""Shared plumbing: errors, HTTP, auth, and typed contracts.

``core/`` is what every public client is built on and is not itself a public
client -- Research lives in ``synth_ai/sdk/research``. Import from the owning
submodule (``synth_ai.core.errors``, ``synth_ai.core.http.retry``, ...); this
package re-exports nothing.
"""
