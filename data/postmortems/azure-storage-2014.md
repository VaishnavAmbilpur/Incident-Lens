# Update on Azure Storage service interruption

| Field | Value |
|---|---|
| Organization | Microsoft Azure |
| Date | 2014-11-19 |
| System | Azure Storage |
| Failure category | config |
| Source | https://azure.microsoft.com/en-us/blog/update-on-azure-storage-service-interruption/ |
| Words | 460 |

---

Update: 11/22/2014, 12:41 PM PST

Since Wednesday, we have been working to help a subset of customers take final steps to fully recover from Tuesday’s storage service interruption. The incident has now been resolved and we are seeing normal activity in the system. You can find updates on the status dashboard:

https://azure.microsoft.com/en-us/status

. If you feel you are still having issues due to the incident, please contact

azcommsm@microsoft.com

, and we will be happy to assist, whether you have a support contract or not. Thank you all again for your feedback regarding communications around this incident. We are actively working to incorporate that feedback into our planning going forward.

Wednesday, November, 19, 2014

As part of a performance update to Azure Storage, an issue was discovered that resulted in reduced capacity across services utilizing Azure Storage, including Virtual Machines, Visual Studio Online, Websites, Search and other Microsoft services. Prior to applying the performance update, it had been tested over several weeks in a subset of our customer-facing storage service for Azure Tables. We typically call this “flighting,” as we work to identify issues before we broadly deploy any updates. The flighting test demonstrated a notable performance improvement and we proceeded to deploy the update across the storage service. During the rollout we discovered an issue that resulted in storage blob front ends going into an infinite loop, which had gone undetected during flighting. The net result was an inability for the front ends to take on further traffic, which in turn caused other services built on top to experience issues.

Once we detected this issue, the change was rolled back promptly, but a restart of the storage front ends was required in order to fully undo the update. Once the mitigation steps were deployed, most of our customers started seeing the availability improvement across the affected regions. While services are generally back online, a limited subset of customers are still experiencing intermittent issues, and our engineering and support teams are actively engaged to help customers through this time.

When we have an incident like this, our main focus is rapid time to recovery for our customers, but we also work to closely examine what went wrong and ensure it never happens again.  We will continually work to improve our customers’ experiences on our platform.  We will update this blog with a RCA (root cause analysis) to ensure customers understand how we have addressed the issue and the improvements we will make going forward.

Update to Azure Customers:

We will continue to investigate what led to this event and will drive the needed improvements to avoid similar situations in the future. In the meantime, we believe it’s important to share a current understanding of the status and the gaps we’ve discovered.
