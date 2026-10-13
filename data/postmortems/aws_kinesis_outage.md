# Summary of the Amazon Kinesis Event in the Northern Virginia (US-EAST-1) Region

**Published:** November 28, 2020  
**Source:** https://aws.amazon.com/message/11201/  

---

On November 25th, 2020, Amazon Kinesis experienced a significant disruption in the US-EAST-1 Region. This message provides additional details on the event, its root cause, and the steps we are taking to prevent recurrence.

## The Event

At 5:15 AM PST on November 25th, alarms began triggering in Amazon Kinesis Data Streams in the US-EAST-1 Region. The issue resulted in elevated error rates and latencies for Kinesis APIs (PutRecord, PutRecords, GetRecords) and impacted numerous dependent AWS services in the region, including Amazon CloudWatch, Amazon Cognito, AWS Lambda, AWS IoT, and parts of the AWS Management Console.

Full recovery for Amazon Kinesis Data Streams occurred at 10:23 PM PST on November 25th, and dependent services recovered shortly thereafter.

## Technical Root Cause Analysis

Amazon Kinesis Data Streams is built with three distinct layers:
1. **Front-End Fleet:** Manages incoming API requests, handles authentication, rate limiting, and request routing.
2. **Back-End Clusters:** Stores streaming data in partitioned shards across distributed storage.
3. **Internal Coordination Layer:** Maintains global cluster topology and shard maps.

### The Trigger: Capacity Addition
Between 2:44 AM and 3:47 AM PST on November 25th, our automated systems added capacity to the Kinesis front-end fleet to support anticipated holiday traffic growth. The capacity addition was relatively small—a few hundred servers added to a front-end fleet that already numbered thousands of nodes.

### The Failure Mechanism: OS Thread Exhaustion
To route incoming requests to the appropriate back-end clusters and shards, every front-end server maintains an in-memory cache of the global shard map. In Kinesis’s internal design, each front-end server maintains active peer-to-peer communication threads with every other server in the fleet to continuously synchronize changes in shard assignment.

Each of these communication channels consumes an operating system thread on the host machine. As the capacity addition crossed a specific fleet-size threshold, the total number of threads required by the front-end process exceeded the operating system's configured maximum user processes limit (`nproc` / thread limit) on each host.

When this limit was reached:
- New internal worker threads could not be spawned.
- Servers began failing health checks as internal thread creation threw resource allocation exceptions.
- The in-memory shard-map cache became unable to refresh, resulting in front-end nodes being unable to route incoming PutRecord and GetRecords requests to back-end shards.
- The failure to route requests cascaded, causing widespread socket timeouts and connection resets for clients.

## Recovery Effort

Mitigating the event required restoring the front-end fleet to a known-healthy operational state. However, recovering the fleet presented significant technical challenges:

1. **Connection Storms on Reboot:** When a front-end server starts up, it must establish communication channels with all other active nodes in the fleet and load the full shard map from the back-end coordination service. 
2. **Cascading Startup Starvation:** If thousands of servers restart simultaneously, the massive inbound connection storm causes all nodes to immediately exhaust their thread allocations and crash during startup.
3. **Controlled Rolling Restarts:** To safely restart without connection storms, engineers had to orchestrate a slow, throttled rolling restart of the fleet in controlled batches across availability zones. Each batch had to be stabilized before introducing subsequent nodes.

By 10:23 PM PST, all front-end nodes had successfully initialized, synchronized shard maps, and resumed processing full production workloads.

## Downstream Services Impact

Because Amazon Kinesis Data Streams is a foundational streaming service in AWS, the degradation cascaded into several downstream services:
- **Amazon CloudWatch:** Ingestion of custom metrics and metric alarms experienced delays because CloudWatch uses Kinesis to buffer metric data before aggregation.
- **Amazon Cognito:** User authentication and token issuance calls were degraded because Cognito writes user login event streams to Kinesis.
- **AWS Lambda:** Event source mappings consuming from Kinesis streams experienced processing delays.
- **AWS Management Console:** Service Health Dashboard (SHD) and CloudWatch dashboard views were intermittently delayed in reflecting real-time status due to metrics pipeline backlogs.

## Preventive Architectural Measures

We have taken and are taking the following actions to ensure this failure mode cannot recur:

1. **Cellular Architecture for Front-End Fleets:** We are decomposing the Kinesis front-end fleet into cellular partitions. In a cellular architecture, front-end servers only communicate with a small, fixed subset of servers within their isolated cell, removing the $O(N^2)$ thread scaling constraint.
2. **OS Configuration & Thread Pooling:** We have increased the operating system thread limits across all front-end server types and transitioned thread management to non-blocking asynchronous event loops (epoll).
3. **Decoupled Status Reporting:** We are accelerating the decoupling of external health reporting tools from the underlying data plane services they monitor.