# Local PNG delivery check · 2026-10-04

The user reported that the browser Download button had no effect. Earlier backend file bindings and 62 passing tests did not establish delivery through that client. Browser-tab inspection timed out; native Codex access was explicitly denied and was not bypassed. No exact browser cause or client fix is claimed.

## Implemented path

**Save PNG to Downloads** is an ordinary Streamlit button. On an explicit click, the local application writes the current rendered PNG to the server user's `~/Downloads`, then shows its absolute path or a write error. This is appropriate for the present local Mac deployment; on a remote server it would save to that server, not the browser's computer. It does not save automatically or extract the live user's session through diagnostic code.

The helper checks the PNG signature and a plain `.png` filename. Exclusive creation refuses overwriting; an identical regular file is reused, while conflicting files and symlinks produce an error without a success message. Saved-result status is keyed by the actual PNG hash and reset on source/workflow changes. Filenames include recipe, strength (`s100`, without a percent sign), dimensions and output hash. Withdrawn AI results offer only a saved original preview.

The separate prepared browser PNG/JSON route remains available. Its export UI identity is now `png-local-delivery-v4`; the engine, PP3 controls, source pixels, strength calculation and model are unchanged by this delivery correction.

## Verification

The temporary-home AppTest exercises the actual current-result button and disk write: no file before a click; a valid 1801×39 PNG exactly matching the displayed/renderer bytes; visible actual path; repeat-click reuse with unchanged modification time; a distinct matching file after a 50% strength change; and conflict rejection with the unrelated file preserved and no success message. Additional checks reject non-PNG bytes, traversal names and symlinks. No real user Downloads file is created by these tests.

Focused delivery and existing display/export checks passed: 2 tests in 2.24 seconds. Full regression: **63 passed in 78.11 seconds**, with the existing optional Matplotlib warning. The recorded JUnit run and source/preservation hashes are in `verification.json`. Exact previous app/test sources are retained as `streamlit_app-before.py` and `tests-before.py`.

## Outstanding acceptance

The actual current user's Save click and browser delivery remain unobserved. The agent has not saved the current personal photo. Server health is not a UI/download acceptance check. This fixes a delivery option and does not establish improved photographic quality; the user's subsequent mountain/lake screenshot reports an unacceptable shadow result and requires a separate quality investigation on the original JPEG.
