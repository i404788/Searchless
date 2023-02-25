from searchless import LinearFloatProxy, resolve_proxies

alpha = LinearFloatProxy(0.5, 0.9, 4)
beta = LinearFloatProxy(0.9, 0.999, 10)

resolve_proxies()

print(f'{alpha=} ({alpha+0.}) {beta=} ({beta+0.})')
