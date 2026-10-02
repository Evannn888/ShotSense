# Historical Snapshot Provenance

Original immutable snapshots contain Chinese UI/documentation from earlier stages. The original screenshot records the initial Chinese interface. They are retained unchanged locally and in earlier Git history, rather than modified and presented as original evidence. Current English documentation preserves the milestones, metrics, and limitations.

| Original path | SHA256 | Original Git revision |
|---|---|---|
| artifacts/model/source_snapshot.zip | `9b0acab1247fea0a5664f6dcf177b2ef3c7112b9ca2f7b838b357341f1004432` | `4c79d10` |
| artifacts/preview_v2/source_snapshot.zip | `08d00eb33f2956c8ad24b140ecb6df48d0b57232574da80af226d0bce7d2d16c` | `4c79d10` |
| artifacts/preview_v3/source_snapshot.zip | `709dea648f919e3a53ccc682e0ef45a175c9535410ee6b12c8376cb08ac3d56f` | `4c79d10` |
| artifacts/model/app-preview.jpg | `b68211c00e05951893558eff817eeddf1a6e3908a503b4ac766bc38f467af4c7` | `4c79d10` |

Retrieve an original from history with `git show 4c79d10:artifacts/model/source_snapshot.zip > /tmp/shotsense-original-model-snapshot.zip`, substituting the desired path.

The new English snapshot is a separate localization artifact, not the original source snapshot for the model/preview acceptance. The production ONNX and checkpoint bytes are unchanged. See `snapshot_provenance.json` for exact sizes and identities.
