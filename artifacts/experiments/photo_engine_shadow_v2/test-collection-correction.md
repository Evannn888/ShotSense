# Test collection correction

The first full run collected the immutable pre-change test source because its filename began test_. It ran68 checks instead of61: current tests passed, but one historical UX test depended on the superseded pending_gentle session key and failed against the new callback. Preserve the99.71s result locally as photo-engine-shadow-v2-collected-snapshot-tests.xml. Rename the historical source to tests-before.py without changing bytes; do not change production behavior or current tests to satisfy a historical snapshot. Re-run normal full discovery.
