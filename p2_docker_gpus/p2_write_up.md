Layer files:

layer/usr/include/x86_64-linux-gnu/cudnn_graph_v9.h
layer/usr/include/x86_64-linux-gnu/cudnn_v9.h
layer/usr/include/x86_64-linux-gnu/cudnn_adv_v9.h
layer/usr/include/x86_64-linux-gnu/cudnn_backend_v9.h
layer/usr/include/x86_64-linux-gnu/cudnn_ops_v9.h
layer/usr/include/x86_64-linux-gnu/cudnn_cnn_v9.h
layer/usr/include/x86_64-linux-gnu/cudnn_version_v9.h
layer/usr/lib/x86_64-linux-gnu/libcudnn.so.9.7.0
layer/usr/lib/x86_64-linux-gnu/libcudnn_engines_runtime_compiled.so.9.7.0
layer/usr/lib/x86_64-linux-gnu/libcudnn_engines_precompiled.so.9.7.0
layer/usr/lib/x86_64-linux-gnu/libcudnn_adv.so.9.7.0
layer/usr/lib/x86_64-linux-gnu/libcudnn_heuristic.so.9.7.0
layer/usr/lib/x86_64-linux-gnu/libcudnn_graph.so.9.7.0
layer/usr/lib/x86_64-linux-gnu/libcudnn_ops.so.9.7.0
layer/usr/lib/x86_64-linux-gnu/libcudnn_cnn.so.9.7.0
layer/usr/share/lintian/overrides/libcudnn9-dev-cuda-12
layer/usr/share/lintian/overrides/libcudnn9-cuda-12
layer/usr/share/doc/libcudnn9-dev-cuda-12/changelog.Debian.gz
layer/usr/share/doc/libcudnn9-dev-cuda-12/copyright
layer/usr/share/doc/libcudnn9-cuda-12/changelog.Debian.gz
layer/usr/share/doc/libcudnn9-cuda-12/copyright
layer/etc/ld.so.cache
layer/var/cache/ldconfig/aux-cache
layer/var/lib/apt/extended_states
layer/var/lib/dpkg/triggers/Lock
layer/var/lib/dpkg/triggers/Unincorp
layer/var/lib/dpkg/lock
layer/var/lib/dpkg/status-old
layer/var/lib/dpkg/lock-frontend
layer/var/lib/dpkg/status
layer/var/lib/dpkg/info/libcudnn9-dev-cuda-12.md5sums
layer/var/lib/dpkg/info/libcudnn9-dev-cuda-12.list
layer/var/lib/dpkg/info/libcudnn9-dev-cuda-12.postinst
layer/var/lib/dpkg/info/libcudnn9-cuda-12.list
layer/var/lib/dpkg/info/libcudnn9-cuda-12.md5sums
layer/var/lib/dpkg/info/libcudnn9-dev-cuda-12.prerm
layer/var/lib/dpkg/alternatives/libcudnn
layer/var/log/apt/eipp.log.xz
layer/var/log/apt/history.log
layer/var/log/apt/term.log
layer/var/log/alternatives.log
layer/var/log/dpkg.log

Layer size: 1.0G

Value of LD_LIBRARY_PATH variable: /usr/local/nvidia/lib:/usr/local/nvidia/lib64

libcufft is present by default, but libcuda is not present by default.

The role of Nvidia Container Toolkit:
- Nvidia container toolkit is used to bridge the gap between containers and the host GPU.
- The toolkit most likely mounts the host's Nvidia drivers into the container runtime.
- The container image I was working with includes cuda toolkit with libraries like libcufft, but it does not include proprietary drivers. It uses the host's drivers instead.
- The toolkit configures environment variables and device file mappings to make sure that containers run smoothly