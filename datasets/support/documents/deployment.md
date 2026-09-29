# Deployment releases

Deployments use a rolling update with at most one unavailable instance. A failed health check triggers an automatic rollback to the previous release. The health check has a 30-second timeout. Configuration changes require a new deployment to take effect.
