============
Contributing
============

Contributions are welcome! Please submit your pull-request, issue or comment
in the `GitHub repo <https://github.com/PyLops/pylops>`_.
You are also welcome to join the `PyLops slack channel <https://pylops.slack.com/>`_.

Instructions for developers
===========================

Installation
~~~~~~~~~~~~
TODO

Contributing documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Documentation is built locally witht the help of the ``Makefile`` in the root
of the project. Use ``make doc`` to build the docs, ``make docupdate`` to
build the docs without cleaning the build directories, and ``make watchdoc``
to rerun ``make docupdate`` at every change in files in ``curvelops/`` or
``docssrc/source/``.

Serve the documents locally with ``make servedocs``.

..  warning::

    PRs changing documentation should be submitted to the
    `gh-pages <https://github.com/PyLops/curvelops/tree/gh-pages>`__ branch.
