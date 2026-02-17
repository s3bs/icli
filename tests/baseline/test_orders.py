"""Characterisation tests for orders.py — order type factory.

Focus: business logic, validation, IBKR quirks, dispatch completeness.
Not testing: "does limit() return orderType='LMT'" — that's testing a string assignment.
"""

import pytest
from decimal import Decimal

ib_async = pytest.importorskip("ib_async")

from ib_async import Order, TagValue
from icli.orders import IOrder, CLIOrderType, markOrderNotGuaranteed


class TestOrderBusinessLogic:
    """Tests that verify actual decision-making in the order factory."""

    def test_stop_limit_falls_back_to_aux_when_no_limit(self):
        """If lmt=0, stopLimit uses aux as the limit price. This is a real
        fallback that someone will accidentally rely on."""
        io = IOrder(action="SELL", qty=50, lmt=0, aux=Decimal("145.00"))
        order = io.stopLimit()
        assert order.lmtPrice == Decimal("145.00")

    def test_stop_with_protection_accepts_lmt_as_aux(self):
        """STP PRT uses aux OR lmt as the stop price — interop convenience."""
        io = IOrder(action="SELL", qty=5, lmt=Decimal("5800.00"), aux=0)
        order = io.stopWithProtection()
        assert order.auxPrice == Decimal("5800.00")

    def test_trailing_stop_rejects_both_aux_and_percent(self):
        """Can't have points AND percent trailing — IBKR API would error."""
        io = IOrder(action="SELL", qty=100, aux=Decimal("2.00"),
                    trailStopPrice=Decimal("148.00"), trailingPercent=Decimal("5.0"))
        with pytest.raises(ValueError, match="Can't specify both"):
            io.trailingStopLimit()

    def test_trailing_stop_by_points_uses_aux(self):
        io = IOrder(action="SELL", qty=100, aux=Decimal("2.00"),
                    trailStopPrice=Decimal("148.00"), trailingPercent=Decimal("0"))
        order = io.trailingStopLimit()
        assert order.auxPrice == Decimal("2.00")
        # When using point-based trailing, trailingPercent is UNSET (not 0)

    def test_trailing_stop_by_percent_uses_percent(self):
        io = IOrder(action="SELL", qty=100, aux=0,
                    trailStopPrice=Decimal("148.00"), trailingPercent=Decimal("5.0"))
        order = io.trailingStopLimit()
        assert order.trailingPercent == Decimal("5.0")

    def test_cash_quantity_zeroes_shares_sets_cashqty(self):
        """The '$5000' string syntax switches from share count to dollar amount."""
        io = IOrder(action="BUY", qty="$5000")
        order = io.market()
        assert order.totalQuantity == 0
        assert order.cashQty == 5000.0

    def test_config_dict_populates_fields(self):
        """Config dict overrides should work for any valid IOrder field."""
        io = IOrder(action="BUY", qty=100, config={"tif": "DAY", "outsiderth": False})
        assert io.tif == "DAY"
        assert io.outsiderth is False


class TestIBKRQuirks:
    """Tests for IBKR-specific behaviour that would be wrong to change."""

    def test_midprice_forces_day_tif(self):
        """MIDPRICE is RTH-only, must be DAY."""
        io = IOrder(action="BUY", qty=100, lmt=Decimal("150.00"), tif="GTC")
        order = io.midprice()
        assert order.tif == "DAY"

    def test_moo_forces_opg_tif_and_no_rth(self):
        """Market-on-Open: tif=OPG, outsideRth=False."""
        io = IOrder(action="BUY", qty=100, outsiderth=True)
        order = io.moo()
        assert order.tif == "OPG"
        assert order.outsideRth is False

    def test_moc_clears_tif_and_no_rth(self):
        """Market-on-Close: tif='', outsideRth=False."""
        io = IOrder(action="SELL", qty=100)
        order = io.moc()
        assert order.tif == ""
        assert order.outsideRth is False

    def test_peg_midpoint_is_notheld(self):
        """PEG MID must be notHeld for IBKRATS dark pool routing."""
        io = IOrder(action="BUY", qty=100, lmt=Decimal("150.00"))
        order = io.pegToMidpoint()
        assert order.notHeld is True

    def test_adaptive_orders_force_day_tif(self):
        """Adaptive algos can't be GTC — IBKR rejects them."""
        for method_name in ["adaptiveSlowLmt", "adaptiveFastLmt", "adaptiveSlowMkt", "adaptiveFastMkt"]:
            io = IOrder(action="BUY", qty=100, lmt=Decimal("150.00"), tif="GTC")
            method = getattr(io, method_name)
            order = method()
            assert order.tif == "DAY", f"{method_name} should force DAY, got {order.tif}"

    def test_adaptive_slow_is_patient_fast_is_urgent(self):
        """Verify the algo param naming isn't swapped."""
        slow = IOrder(action="BUY", qty=100, lmt=Decimal("150")).adaptiveSlowLmt()
        fast = IOrder(action="BUY", qty=100, lmt=Decimal("150")).adaptiveFastLmt()
        slow_prio = next(tv.value for tv in slow.algoParams if tv.tag == "adaptivePriority")
        fast_prio = next(tv.value for tv in fast.algoParams if tv.tag == "adaptivePriority")
        assert slow_prio == "Patient"
        assert fast_prio == "Urgent"

    def test_default_trigger_is_midpoint(self):
        """Trigger method 8 = midpoint. Changing this default would affect all orders."""
        io = IOrder(action="BUY", qty=100)
        assert io.trigger == 8


class TestDispatchCompleteness:
    """Verify the dispatch table is complete and consistent."""

    def test_all_dispatch_keys_produce_orders(self):
        """Every entry in the omap must produce a non-None Order."""
        known_types = [
            "LMT", "MKT", "STP", "STP LMT", "LIT", "MIT", "REL",
            "TRAIL LIMIT", "MIDPRICE", "SNAP MID", "SNAP MKT", "SNAP PRIM",
            "LMT + ADAPTIVE + SLOW", "LMT + ADAPTIVE + FAST",
            "MKT + ADAPTIVE + SLOW", "MKT + ADAPTIVE + FAST",
            "MKT PRT", "MTL", "STP PRT", "PEG MID",
            "REL + MKT", "REL + LMT", "LMT + MKT",
            "MOO", "MOC",
        ]
        for otype in known_types:
            io = IOrder(action="BUY", qty=100, lmt=Decimal("150.00"), aux=Decimal("145.00"),
                        trailStopPrice=Decimal("148.00"), trailingPercent=Decimal("0"))
            order = io.order(otype)
            assert order is not None, f"Dispatch failed for: {otype}"

    def test_invalid_type_returns_none(self):
        io = IOrder(action="BUY", qty=100)
        assert io.order("FAKE_ORDER_TYPE") is None

    def test_algomap_keys_all_resolve(self):
        """Every user-facing abbreviation in ALGOMAP must map to a
        dispatchable order type."""
        # Python 3.12+ supports PEP 695 syntax (type BuySell = ...).
        # Skip this test on earlier Python versions where helpers.py won't import.
        try:
            from icli.helpers import ALGOMAP
        except SyntaxError:
            pytest.skip("ALGOMAP requires Python 3.12+ (PEP 695 syntax)")

        for alias, ibkr_name in ALGOMAP.items():
            io = IOrder(action="BUY", qty=100, lmt=Decimal("150.00"), aux=Decimal("145.00"),
                        trailStopPrice=Decimal("148.00"), trailingPercent=Decimal("0"))
            order = io.order(ibkr_name)
            assert order is not None, \
                f"ALGOMAP['{alias}'] -> '{ibkr_name}' does not dispatch to a valid order"


class TestMarkOrderNotGuaranteed:
    def test_adds_required_combo_tags(self):
        order = Order()
        order.smartComboRoutingParams = []
        markOrderNotGuaranteed(order)
        tags = {tv.tag: tv.value for tv in order.smartComboRoutingParams}
        assert tags == {"NonGuaranteed": "1", "LeginPrio": "0", "DontLeginNext": "1"}
