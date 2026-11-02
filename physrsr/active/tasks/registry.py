"""Task selection belongs to application composition, not the prediction engine."""
from ..core.factories import load_factory

BUILTIN_ADAPTERS = {
    "rod_lift": "physrsr.active.tasks.rod_lift:RodMeasuredAdapter",
    "three_rod_lift_return": "physrsr.active.tasks.rod_scene:ThreeRodAdapter",
}
DEFAULT_TASKS = ("rod_lift", "three_rod_lift_return")


def create_adapters(specifications=None):
    """Accept built-in task names or explicit MODULE:FACTORY extensions."""
    specifications = DEFAULT_TASKS if specifications is None else specifications
    return [load_factory(BUILTIN_ADAPTERS.get(spec, spec)) for spec in specifications]
