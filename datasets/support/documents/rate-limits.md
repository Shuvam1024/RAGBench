# Request quotas

The standard API quota is 120 requests per minute per workspace. Exceeding the quota returns HTTP 429 and a Retry-After header. Clients should wait for the indicated delay and use exponential backoff. Batch imports have a separate quota of 10 jobs per hour.
