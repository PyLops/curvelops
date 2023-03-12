============
Contributing
============

Contributions are welcome! Please submit your pull-request, issue or comment
in the `GitHub repo <https://github.com/PyLops/pylops>`__. You are also
welcome to join the `PyLops slack channel <https://pylops.slack.com/>`__.

Installation for developers
===========================

Developers should clone the
`main <https://github.com/PyLops/curvelops/tree/main>`__ branch of the
repository and install the dev requiments:

..  code-block:: console

    $ git clone https://github.com/PyLops/curvelops
    $ git remote add upstream https://github.com/PyLops/curvelops
    $ make dev-install

They should then follow the same instructions in the :ref:`Requirements`
section. We recommend installing dependencies into a separate environment.
Finally, they can install Curvelops with

..  code-block:: console

    $ python3 -m pip install -e .

Developers should also install `pre-commit <https://pre-commit.com/>`__ hooks with

..  code-block:: console

    $ pre-commit install


Developer workflow
==================

Developers should start from a fresh copy of main with

..  code-block:: console

    $ git checkout main
    $ git pull upstream main

Before you start making changes, create a new branch with

..  code-block:: console

    $ git checkout -b patch-some-cool-feature

After implementing your cool feature (including tests 🤩), commit your changes
to kick-off the pre-commit hooks. These will reject and "fix" your code by
running the proper hooks. At this point, the user must check the changes and
then stage them before trying to commit again.

Once changes are committed, we encourage developers to lint, check types and
build/check documentation with:

..  code-block:: console

    $ make tests
    $ make lint
    $ make typeannot
    $ make coverage
    $ make doc
    $ make servedoc

Once everything is in order, and your code has been pushed to GitHub,
navigate to https://github.com/PyLops/curvelops and submit your PR!

..  warning::

    PRs changing documentation should be submitted to the
    `gh-pages <https://github.com/PyLops/curvelops/tree/gh-pages>`__ branch.

Contributing documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Documentation is built locally with the help of the ``Makefile`` in the root
of the project. Use ``make doc`` to build the docs, ``make docupdate`` to
build the docs without cleaning the build directories, and ``make watchdoc``
to rerun ``make docupdate`` at every change in files in ``curvelops/`` or
``docssrc/source/``. Serve the documents locally with ``make servedocs``.
PRs changing documentation should be submitted to the `gh-pages <https://github.com/PyLops/curvelops/tree/gh-pages>`__ branch.
