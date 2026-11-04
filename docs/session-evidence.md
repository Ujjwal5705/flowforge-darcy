# Session evidence

Application: QUEST Lab Full Stack AI Pre-doctoral application, Ujjwal Sharma, 2026-11-03.

- First user-message timestamp in this session: **2026-11-03T16:55:35.582Z** (22:25:35.582 IST).
- Hard final-commit deadline: **2026-11-06T16:55:35.582Z** (22:25:35.582 IST).
- Session ID: `01a1a256-5f99-7313-8561-24ecaac40350`.
- Observed model in the session's `turn_context`: `gpt-6.1-sol`.
- Raw log: `/Users/ujjwalsharma/.codex/sessions/2026/11/03/rollout-2026-11-03T22-25-32-01a1a256-5f99-7313-8561-24ecaac40350.jsonl`.
- The request was supplied as an attached text file with the requested header. The raw chat contains the attachment wrapper; it has not been changed to alter that fact.
- No other agents, models, or development sessions are used for this project.
- The raw log is read only and must be submitted unedited. This document is a project note, not a replacement log.

## Initial inspection

The workspace was empty and had no Git repository or remote. Hardware: macOS 15.0.1,
ARM64, eight logical CPUs, 8 GiB unified RAM, approximately 61.8 GB available disk.
The pre-existing Python 3.8 environment is unsuitable for the current package, so
the project uses an isolated Python 3.11.15 environment. Node 22.17.0 and npm 10.9.2
are available. No CUDA GPU. The initial sandboxed PyTorch reported MPS unavailable;
CPU is the conservative compute target. Docker Desktop reports a Linux ARM64 daemon
with about 4.1 GB memory when accessed outside the filesystem sandbox.

An existing Hugging Face token authenticates as `ujjwal75` with write access.
No token value was displayed. Publishing a new public Space requires an explicit
audience decision after the application is ready for review. No paid resources
will be provisioned.

## Final evidence

The owner explicitly approved uploading the reviewed source/checkpoint/metadata
and six CC BY 4.0 examples to `ujjwal75/flowforge-darcy-2026`, followed by public
release after verification. Private CPU/CUDA checks passed before publication;
the final anonymous browser suite passed six tests. Raw training data and this
session log were excluded from the Space snapshot. No paid upgrade was requested.

A final read of this raw log still identified the same session ID and only
`gpt-6.1-sol` model entries. The attachment-wrapper caveat above remains unchanged.
The final commit is made before the 72-hour deadline; actual timestamps remain in
Git and the original log, without rewriting history.

After the final commit, a portable Git bundle is prepared and verified locally:

```bash
git bundle create artifacts/submission/flowforge.bundle main
git bundle verify artifacts/submission/flowforge.bundle
```

It preserves the application branch and its complete commit history, excluding
internal Codex capture refs, ignored datasets, caches and
the external raw session log. A reviewer can restore it with
`git clone --branch main flowforge.bundle flowforge`. Submit the exact original `.jsonl` file
separately, unedited; this project note and bundle do not replace it.
