




Saturn RTL sim

```scala
class REFV128D64ShuttleTapeoutConfig extends Config(
  new chipyard.config.WithSystemBusWidth(128) ++
  new saturn.shuttle.WithShuttleVectorUnit(128, 64, VectorParams.refParams) ++
  new shuttle.common.WithShuttleTileBeatBytes(16) ++
  new shuttle.common.WithTCM ++
  new shuttle.common.WithNShuttleCores(1) ++
  new chipyard.config.AbstractConfig)

```




