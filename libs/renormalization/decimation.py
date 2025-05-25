from libs.models.types import HasModelAttribute


class Decimation:
    def __init__(
        self,
        l,  # decimation shrink l, i.e. 2, for half decimation, and so.
        index_set_map,  # fn from input idx to output idx map
        neighbors_map,  # fn from input neighbors to output neighbors map
    ):
        self.l = l
        self.index_set_map = index_set_map
        self.neighbors_map = neighbors_map

    def decimation(model: HasModelAttribute):
        # I need to think of this more to firmly understand which is the better choice.
        # TODO This is currently busy node. src/dev for current branch is seeking for structure.
        return
