import argparse

import pytest

from drive import DryRunDrive, TankDrive, parse_ports


class FakeBrain:
    def __init__(self):
        self.calls = []

    def spin(self, port, pct):
        self.calls.append(("spin", port, pct))

    def stop(self, port="ALL", mode="BRAKE"):
        self.calls.append(("stop", port, mode))

    def close(self):
        self.calls.append(("close",))


def test_parse_ports():
    assert parse_ports("1,2") == [1, 2]
    assert parse_ports("-3, -4") == [-3, -4]
    for bad in ("", "a,b", "0", "22", "1,-22"):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_ports(bad)


def test_reversed_ports_get_negated_speed():
    brain = FakeBrain()
    TankDrive(brain, [1, 2], [-3, -4]).set(30, 20)
    assert brain.calls == [("spin", 1, 30), ("spin", 2, 30), ("spin", 3, -20), ("spin", 4, -20)]


def test_unchanged_command_not_resent():
    brain = FakeBrain()
    d = TankDrive(brain, [1], [2])
    d.set(30, 30)
    d.set(30.01, 29.99)
    assert len(brain.calls) == 2


def test_zero_means_brake_and_close_stops_then_closes():
    brain = FakeBrain()
    d = TankDrive(brain, [1], [2])
    d.set(30, 30)
    d.set(0, 0)
    assert brain.calls[-1] == ("stop", "ALL", "BRAKE")
    d.close()
    assert brain.calls[-2:] == [("stop", "ALL", "BRAKE"), ("close",)]


def test_failed_send_is_retried():
    class Flaky(FakeBrain):
        fail = True

        def spin(self, port, pct):
            if self.fail:
                self.fail = False
                raise RuntimeError("serial hiccup")
            super().spin(port, pct)

    brain = Flaky()
    d = TankDrive(brain, [1], [2])
    with pytest.raises(RuntimeError):
        d.set(30, 30)
    d.set(30, 30)
    assert ("spin", 1, 30) in brain.calls


def test_port_on_both_sides_rejected():
    with pytest.raises(ValueError):
        TankDrive(FakeBrain(), [1, 2], [-2, 3])


def test_dry_run_prints_changes_only(capsys):
    d = DryRunDrive()
    d.set(10, 10)
    d.set(10, 10)
    d.stop()
    assert capsys.readouterr().out.count("[dry-run]") == 2
