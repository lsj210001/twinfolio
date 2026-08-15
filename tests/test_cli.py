import pytest

from book_translate.cli import main


def test_cli_test_num_must_be_positive():
    assert main(["book.epub", "--test-num", "0"]) == 2


def test_cli_help_lists_new_flags(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["-h"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--test" in out
    assert "--only" in out
    assert "--retranslate" in out
    assert "--use-context" in out
