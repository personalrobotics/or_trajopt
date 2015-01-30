# or_trajopt #

OpenRAVE prpy bindings for TrajOpt.

This plugin downloads and builds TrajOpt from source and creates a
Python wrapper that interfaces it with the prpy planning API.

All of the normal TrajOpt python bindings are also available when this
package is included in a catkin workspace.

## Dependencies ##
```
$ sudo apt-get install libopenscenegraph-dev cmake libboost-all-dev libeigen3-dev python-numpy
```
(See http://rll.berkeley.edu/trajopt/doc/sphinx_build/html/install.html#dependencies)

## License ##
See [LICENSE.txt](LICENSE.txt).

## OpenSceneGraph issue in Parallels VM ##

You will likely receive several "undefined reference to
`glXCreateGLXPbufferSGIX`" linking errors when building TrajOpt inside a
Parallels virtual machine. This is [known issue](https://forum.parallels.com/threads/glxcreateglxpbuffersgix-missing-from-parallels-tools-libgl-so.257728/)
that results from OpenSceneGraph using a deprecated OpenGL command that is not
implemented by Parallels. You can work around this issue by building
OpenSceneGraph from source  the `GLX_SGIX_pbuffer` pre-processor define
`#undef`ed.

If you are on Ubuntu, you can easily build a patched version of OpenSceneGraph
using git-buildpackage. To do so, run:

    $ git clone https://github.com/mkoval/openscenegraph-release.git
    $ cd openscenegraph-release
    $ gbp buildpackage --git-debian-branch=master -uc -us

This will produce several `.deb` packages in the parent directory. You can
install these Debian packages using your favorite installation method (`gdebi`,
`dpkg -i`, etc).

## Authors ##
* Pras Velagapudi `<pkv@cs.cmu.edu>`
