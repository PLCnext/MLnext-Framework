This directory contains "newsfragments" which are short files that contain a
small **ReST**-formatted text that will be added to the next ``CHANGELOG``.

The ``CHANGELOG`` will be read by users, so this description should be aimed to
users instead of describing internal changes which are only relevant to the
developers.

Make sure to use full sentences with correct case and punctuation, for
example ::

    Fix issue with bar from the ``foo`` module.

Each file should be named like ``<ISSUE>.<TYPE>.rst``, where ``<ISSUE>`` is an
issue number, and ``<TYPE>`` is one of:

* ``feature``: new user facing features, like new command-line options and new behavior.
* ``change``: changes to existing logic or behavior.
* ``deprecation``: feature deprecation.

For example: ``123.feature.rst``, ``456.change.rst``.

If your PR fixes an issue, use that number here. If there is no issue, then
after you submit the PR and get the PR number you can add a changelog using
that instead.

If you are not sure what issue type to use, don't hesitate to ask in your PR.

``towncrier`` preserves multiple paragraphs and formatting (code blocks, lists,
and so on), but for entries other than ``features`` it is usually better to
stick to a single paragraph to keep it concise. You can install ``towncrier``
and then run ``towncrier --draft`` if you want to get a preview of how your
change will look in the final release notes.
