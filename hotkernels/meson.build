# Build of the hotkernels package: the Python sources, and the extension
# module that binds the kernels to NumPy.
#
# The package's version, and the only place it is written. `uv run release prepare`
# changes it; the release workflow refuses a tag that disagrees with it.
#
# No language is declared here: C is added only where the extension is built,
# so the package builds on any other platform without a compiler.
project('hotkernels', version: '0.3.1', meson_version: '>=1.1',
        default_options: ['buildtype=release'])

fs = import('fs')
py = import('python').find_installation(pure: false)

package = 'hotkernels'
source = 'src' / package

py.install_sources(
  [source / '__init__.py', source / '__main__.py', source / 'environment.py', source / 'hooks.py',
   source / 'report.py'],
  subdir: package,
)
py.install_sources([source / 'templates' / 'report.txt.j2'], subdir: package / 'templates')

# The kernels are assembly for two platforms: ARM64 on macOS and x86-64 on
# Linux. `kernel_arch` is the name of a platform's directory under
# kernels/<name>/. Anywhere else the package installs without the extension,
# and install() reports that it has nothing to do.
kernel_arch = ''
if host_machine.system() == 'darwin' and host_machine.cpu_family() == 'aarch64'
  kernel_arch = 'arm64'
elif host_machine.system() == 'linux' and host_machine.cpu_family() == 'x86_64'
  kernel_arch = 'x86_64'
endif

if kernel_arch != ''
  add_languages('c', native: false, required: true)

  # The kernel set: kernel sources, prototypes, the kernel table and the
  # judge's C flags (see tools/generate.py). A source distribution carries one
  # per architecture in kernel-set/<arch>/, written when it was made. A
  # checkout has none, so the one for this build is written into the build
  # directory from ../kernels.
  if fs.is_dir('kernel-set')
    kernel_set = meson.current_source_dir() / 'kernel-set' / kernel_arch
  else
    generate = files('tools/generate.py')
    kernel_set = meson.current_build_dir() / 'kernel-set'
    run_command(py, [generate, 'write', kernel_set, kernel_arch], check: true)
    # `meson dist` runs this on the distribution it makes, so it carries the set.
    meson.add_dist_script(py, generate, 'dist')
  endif

  # Reads a file of the kernel set. fs.read() refuses files in the build tree,
  # which is where a checkout's kernel set is.
  read_file = [py, '-c', 'import sys; sys.stdout.write(open(sys.argv[1]).read())']

  # Either list can be empty: a platform where every kernel has assembly has
  # no C sources.
  kernel_sources = kernel_set / 'sources'
  asm_sources = []
  foreach name : run_command(read_file + [kernel_set / 'assembly-sources.txt'], check: true).stdout().strip().split('\n')
    if name != ''
      asm_sources += kernel_sources / name
    endif
  endforeach
  c_sources = []
  foreach name : run_command(read_file + [kernel_set / 'c-sources.txt'], check: true).stdout().strip().split('\n')
    if name != ''
      c_sources += kernel_sources / name
    endif
  endforeach

  # The judge compiles for the machine it runs on: -mcpu=native on ARM64,
  # -march=native on x86-64. A wheel is built once for every machine of its
  # platform, so it names a CPU or a level instead; see meson.options.
  judge_c_flags = []
  foreach flag : run_command(read_file + [kernel_set / 'c-flags.txt'], check: true).stdout().strip().split('\n')
    if get_option('kernel_cpu') != 'native' and (flag == '-mcpu=native' or flag == '-march=native')
      judge_c_flags += flag.replace('native', get_option('kernel_cpu'))
    else
      judge_c_flags += flag
    endif
  endforeach

  # A kernel that ships as C is its naive reference, compiled with the flags
  # the judge timed it with. The generated header is force-included, as the
  # judge does, so a signature that disagrees with the spec fails the build.
  kernel_libraries = []
  if c_sources.length() > 0
    kernel_libraries += static_library('c_kernels', c_sources,
      c_args: judge_c_flags + ['-include', kernel_set / 'kernels.h'],
      override_options: ['c_std=c11', 'warning_level=0'],
    )
  endif

  numpy_include = run_command(py, ['-c', 'import numpy; print(numpy.get_include())'], check: true).stdout().strip()
  binding = source / '_binding'
  py.extension_module('_binding',
    [binding / 'module.c', binding / 'hook_slot.c', binding / 'hook_loop.c',
     binding / 'hook_function.c', binding / 'hook_method.c', binding / 'hook_index.c',
     binding / 'hook_bottleneck.c', binding / 'hook_ufunc.c'] + asm_sources,
    include_directories: include_directories(numpy_include, is_system: true),
    c_args: ['-O3', '-I' + kernel_set],
    override_options: ['c_std=c11', 'warning_level=2'],
    link_with: kernel_libraries,
    subdir: package,
    install: true,
  )
  install_data(kernel_set / 'kernels.json', install_dir: py.get_install_dir() / package)
endif
