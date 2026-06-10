# LiteLLM v1.88.1 upgrade (ai-infra)

## Current state

| Item | Value |
|------|-------|
| Upstream base | [v1.88.1](https://github.com/BerriAI/litellm/tree/v1.88.1) |
| Custom branch | `main` |
| Custom tag | `v1.88.1-ai-infra` |
| Pre-upgrade snapshot | branch `backup/pre-v1.88.1-20260610` @ `fc47fb1f49` |

## Revert if something goes wrong

### Option A — reset to pre-upgrade snapshot (fastest)

```bash
cd src/litellm-proxy
git checkout backup/pre-v1.88.1-20260610
make ai-sync
make ai-restart
```

### Option B — undo merge commits on main

```bash
cd src/litellm-proxy
git checkout main
git reset --hard backup/pre-v1.88.1-20260610
make ai-sync
make ai-restart
```

### Option C — restore from stash (includes in-progress OCA removal WIP)

```bash
cd src/litellm-proxy
git stash list   # look for "WIP: OCA removal before v1.88.1 merge"
git checkout backup/pre-v1.88.1-20260610
git stash pop
```

## Customizations kept

| Customization | Still needed? | Notes |
|---------------|---------------|-------|
| Anthropic effort/display passthrough (`utils.py`, adapters) | **Yes** | Upstream added `output_config.effort` but not full `display: omitted/summarized` handling |
| OCI embeddings | **No (upstream)** | v1.88.1 refactored `OCIOEmbedConfig` with shared `sign_oci_request` |
| OCA provider | **Removed** | De-integrated from ai-infra; OCI GenAI covers production models |
| Makefile `ai-*` targets | **Yes** | Updated for OCI + passthrough checks (no OCA) |
| UI `_experimental/out/` renames | **Dropped** | Local build artifacts; replaced by upstream v1.88.1 dashboard build |

## Post-upgrade checklist

```bash
cd src/litellm-proxy
make ai-sync
make ai-check
make ai-restart
# optional smoke test
make ai-verify
```
