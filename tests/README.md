# Tests

To run the tests, you will need to have `pytest` installed. Run the tests like this:

```bash
pytest
```

The suite uses the installed `arcengine` and `arc_agi` types and mocks Arcade in
orchestration tests, so unit tests do not contact the public API. Recording fixtures
use isolated pytest temporary directories instead of deleting a shared directory.

For more information on tests, please see the [tests documentation](https://arcprize.org/docs#testing).
