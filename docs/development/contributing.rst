.. _contributing:

======================================
Reporting issues and proposing changes
======================================

The lkdata module is developed on its own `GitHub repository <https://github.com/lightkurve/lkdata>`

If you encounter a problem with lkdata, we encourage you to
`open an issue on the GitHub repository <https://github.com/lightkurve/lkdata/issues>`_.

If you would like to propose a change or bug fix to lkdata, please go ahead and open a pull request
using the steps explained below.


Proposing changes to Lightkurve modules using GitHub Pull Requests
------------------------------------------------------------------

We welcome suggestions for enhancements or new features to Lightkurve via GitHub.

If you want to make a significant change such as adding a new feature,
we recommend opening a GitHub issue to discuss the changes first.
Once you are ready to propose the changes, please go ahead and open a pull request.

If in doubt on how to open a pull request, we recommend Astropy's
"`How to make a code contribution <http://docs.astropy.org/en/stable/development/workflow/development_workflow.html>`_" tutorial.

After you follow steps 1 through 4 from our `install instructions <development>`_, you can take the following steps to open a pull request to any `Lightkurve` module.


5. Create a new branch
~~~~~~~~~~~~~~~~~~~~~~

You are now ready to start contributing changes.
Before making new changes, always make sure to retrieve the latest version
of the source code as follows:

.. code-block:: bash

    $ git checkout main
    $ git pull upstream main

You are now ready to create your own branch with a name of your choice:

.. code-block:: bash

    $ git branch name-of-your-branch
    $ git checkout name-of-your-branch


6. Make changes and add them to the repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can now go ahead and modify source files.
When you are happy about a change, you can commit it
to your local version of the repository as follows:


.. code-block:: bash

    $ git add FILE-YOU-ADDED-OR-MODIFIED
    $ git commit -m "description of changes"


7. Push your changes to GitHub and open a Pull Request
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, send the changes to the fork of the Lightkurve module that resides in your GitHub account:

.. code-block:: bash

    $ git push origin name-of-your-branch

Head to https://github.com/lightkurve/lkdata after issuing the `git push`
command above. You should automatically see a button that says "Compare and open a pull request".
Click the button and submit your pull request!
