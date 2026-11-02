# External workspace source overlay

Source: https://github.com/ChoujiajunStats/stonefish_slam_ws.git at
`73fd4556f1eb8fc19e70f07fd83b6a1f38cb92c3` plus the local tracked patch and these
10 selected source/config files. Exact hashes are in
[`../provenance/stonefish-workspace-source.json`](../provenance/stonefish-workspace-source.json).

Restore into a **separate checkout** at that pin: first inspect/apply
`../provenance/stonefish-workspace-local.patch`, then copy this directory's
`controller/`, `navigation/`, `runtime/`, `docker/` and `vendor/` into the checkout,
keeping relative paths and file modes. Do not overwrite a running dirty workspace.

No weights, caches, Porth assets, sessions or credentials are included. Prepared
base images and external assets/weights are still required; patch + overlay is
not a complete environment or a blank-machine reproducibility guarantee.
