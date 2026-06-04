from src.server import (
    _fusion_add_keyframe_to_input,
    _fusion_delete_keyframe_from_input,
    _fusion_keyframe_times,
    _fusion_read_keyframes,
    _fusion_read_point_keyframes,
)


class ConnectedOutputStub:
    def __init__(self, tool):
        self.tool = tool

    def GetTool(self):
        return self.tool


class InputStub:
    def __init__(self, data_type="Point", create_on_assign=True, connected_tool=None):
        self.attrs = {"INPS_ID": "Center", "INPS_Name": "Center", "INPS_DataType": data_type}
        self.values = {}
        self.create_on_assign = create_on_assign
        self.connected_tool = connected_tool

    def GetAttrs(self):
        return dict(self.attrs)

    def __setitem__(self, time, value):
        if self.create_on_assign:
            self.values[time] = value
            if self.connected_tool is not None and hasattr(self.connected_tool, "set_point"):
                self.connected_tool.set_point(time, value)

    def __getitem__(self, time):
        return self.values[time]

    def GetKeyFrames(self):
        if self.connected_tool is not None and hasattr(self.connected_tool, "GetKeyFrames"):
            return self.connected_tool.GetKeyFrames()
        return {time: value for time, value in self.values.items()}

    def GetConnectedOutput(self):
        return ConnectedOutputStub(self.connected_tool) if self.connected_tool is not None else None

    def DeleteKeyFrames(self, time):
        self.values.pop(time, None)
        if self.connected_tool is not None and hasattr(self.connected_tool, "DeleteKeyFrames"):
            self.connected_tool.DeleteKeyFrames(time)
        return None


class IndexedScalarInputStub(InputStub):
    def __init__(self):
        super().__init__("Number")
        self.attrs.update({"INPS_ID": "Blend", "INPS_Name": "Blend"})

    def GetKeyFrames(self):
        return {index + 1: time for index, time in enumerate(sorted(self.values))}


class BezierSplineStub:
    def __init__(self, create_on_assign=True):
        self.values = {}
        self.create_on_assign = create_on_assign

    def GetAttrs(self):
        return {"TOOLS_RegID": "BezierSpline", "TOOLS_Name": "Path1Displacement"}

    def __setitem__(self, time, value):
        if self.create_on_assign:
            if isinstance(value, dict):
                value = value.get(1, value.get("1", value.get("Value", value.get("value", 0))))
            self.values[float(time)] = {1: float(value)}

    def GetKeyFrames(self):
        return dict(self.values)

    def DeleteKeyFrames(self, time):
        self.values.pop(float(time), None)
        self.values.pop(time, None)
        return None


class PathStub(InputStub):
    def __init__(self, create_on_assign=True, create_displacement=True):
        super().__init__("Point", create_on_assign=create_on_assign)
        self.displacement_spline = BezierSplineStub(create_on_assign=create_displacement)
        self.Displacement = InputStub("Number", create_on_assign=create_displacement, connected_tool=self.displacement_spline)

    def GetAttrs(self):
        return {"TOOLS_RegID": "Path", "TOOLS_Name": "Path1"}

    def Delete(self):
        return None


class XYPathStub:
    def __init__(self, create_on_assign=True):
        self.x_spline = BezierSplineStub(create_on_assign=create_on_assign)
        self.y_spline = BezierSplineStub(create_on_assign=create_on_assign)
        self.x_input = InputStub("Number", connected_tool=self.x_spline)
        self.x_input.attrs.update({"INPS_ID": "X", "INPS_Name": "X"})
        self.y_input = InputStub("Number", connected_tool=self.y_spline)
        self.y_input.attrs.update({"INPS_ID": "Y", "INPS_Name": "Y"})

    def GetAttrs(self):
        return {"TOOLS_RegID": "XYPath", "TOOLS_Name": "XYPath1"}

    def GetInputList(self):
        return {1: self.x_input, 2: self.y_input}

    def set_point(self, time, value):
        point = value if isinstance(value, dict) else {1: value[0], 2: value[1]}
        self.x_spline[time] = point.get(1, point.get("1"))
        self.y_spline[time] = point.get(2, point.get("2"))

    def GetKeyFrames(self):
        times = sorted(set(self.x_spline.GetKeyFrames()) | set(self.y_spline.GetKeyFrames()))
        return {index + 1: time for index, time in enumerate(times)}

    def DeleteKeyFrames(self, time):
        self.x_spline.DeleteKeyFrames(time)
        self.y_spline.DeleteKeyFrames(time)
        return None

    def Delete(self):
        return None


class CompStub:
    def __init__(self, path):
        self.path = path

    def Path(self, opts=None):
        return self.path


class ToolStub:
    def __init__(self, inp):
        self.inp = inp
        self.path = None
        self.modifiers = []
        self.set_inputs = []

    def __getitem__(self, name):
        return self.inp if name == "Center" else None

    def GetInputList(self):
        return {1: self.inp}

    def AddModifier(self, input_name, modifier):
        self.modifiers.append((input_name, modifier))
        if modifier == "XYPath":
            self.path = XYPathStub(create_on_assign=self.inp.create_on_assign)
            self.inp.connected_tool = self.path
        elif modifier == "Path":
            self.path = PathStub(create_on_assign=self.inp.create_on_assign)
            self.inp.connected_tool = self.path
        return True

    def SetInput(self, input_name, value, time=None):
        self.set_inputs.append((input_name, value, time))
        if time is None and isinstance(value, PathStub):
            self.path = value
            self.inp = value
        return True

    def GetInput(self, input_name, time=None):
        if input_name != "Center":
            return None
        if self.inp.connected_tool is not None and isinstance(self.inp.connected_tool, XYPathStub):
            x_keys = self.inp.connected_tool.x_spline.GetKeyFrames()
            y_keys = self.inp.connected_tool.y_spline.GetKeyFrames()
            if time is not None:
                keys = sorted(set(x_keys) | set(y_keys))
                eligible = [key for key in keys if key <= float(time)]
                key = eligible[-1] if eligible else (keys[0] if keys else None)
                if key is not None:
                    return {1: x_keys.get(key, {1: 0.5})[1], 2: y_keys.get(key, {1: 0.5})[1], 3: 0.0}
        if self.inp.connected_tool is not None and isinstance(self.inp.connected_tool, PathStub):
            path_keys = self.inp.connected_tool.GetKeyFrames()
            if time is not None and time in path_keys:
                value = path_keys[time]
                return {1: value[1], 2: value[2], 3: 0.0}
        if time is not None and time in self.inp.values:
            return self.inp.values[time]
        if self.inp.values:
            return self.inp.values[sorted(self.inp.values)[-1]]
        return None


def test_point_keyframe_adds_xypath_modifier_and_verifies_readback():
    inp = InputStub("Point")
    path = PathStub()
    comp = CompStub(path)
    tool = ToolStub(inp)

    result = _fusion_add_keyframe_to_input(comp, tool, "Center", inp, 12, [0.25, 0.75])

    assert result["success"] is True
    assert ("Center", "XYPath") in tool.modifiers
    assert any(row[0] == "Center" for row in tool.set_inputs)
    assert 12.0 in tool.path.x_spline.GetKeyFrames()
    assert 12.0 in tool.path.y_spline.GetKeyFrames()
    assert result["modifier"] == "xypath"
    assert result["displacement_keyed"] is True


def test_point_keyframe_reuses_existing_xypath_across_sequential_calls():
    inp = InputStub("Point")
    path = PathStub()
    comp = CompStub(path)
    tool = ToolStub(inp)

    first = _fusion_add_keyframe_to_input(comp, tool, "Center", inp, 0, [0.482, 0.5])
    second = _fusion_add_keyframe_to_input(comp, tool, "Center", tool.inp, 12, [0.5, 0.5])

    assert first["success"] is True
    assert second["success"] is True
    assert first["attached_new_path"] is True
    assert second["attached_new_path"] is False
    assert tool.modifiers == [("Center", "XYPath")]
    assert sorted(tool.path.x_spline.GetKeyFrames()) == [0.0, 12.0]
    assert tool.path.x_spline.GetKeyFrames()[0.0][1] == 0.482
    assert tool.path.x_spline.GetKeyFrames()[12.0][1] == 0.5

    keyframes = _fusion_read_point_keyframes(tool, "Center", tool.inp)
    assert [row["time"] for row in keyframes] == [0.0, 12.0]
    assert keyframes[0]["source"] == "XYPath.XY.BezierSpline"


def test_point_keyframe_delete_removes_path_and_displacement_keys():
    inp = InputStub("Point")
    path = PathStub()
    comp = CompStub(path)
    tool = ToolStub(inp)

    _fusion_add_keyframe_to_input(comp, tool, "Center", inp, 0, [0.482, 0.5])
    _fusion_add_keyframe_to_input(comp, tool, "Center", tool.inp, 12, [0.5, 0.5])

    result = _fusion_delete_keyframe_from_input(tool, "Center", tool.inp, 12)

    assert result["success"] is True
    assert 12.0 not in tool.path.x_spline.GetKeyFrames()
    assert 12.0 not in tool.path.y_spline.GetKeyFrames()


def test_point_keyframe_converts_existing_path_to_xypath():
    inp = InputStub("Point")
    old_path = PathStub()
    inp.connected_tool = old_path
    old_path[0] = {1: 0.482, 2: 0.5}
    old_path.displacement_spline[0] = 0.0
    comp = CompStub(PathStub())
    tool = ToolStub(inp)

    result = _fusion_add_keyframe_to_input(comp, tool, "Center", inp, 12, [0.5, 0.5])

    assert result["success"] is True
    assert ("Center", "XYPath") in tool.modifiers
    assert tool.path.GetAttrs()["TOOLS_RegID"] == "XYPath"
    assert sorted(tool.path.x_spline.GetKeyFrames()) == [0.0, 12.0]


def test_point_keyframe_returns_error_when_resolve_does_not_create_key():
    inp = InputStub("Point", create_on_assign=False)
    path = PathStub(create_on_assign=False, create_displacement=False)
    comp = CompStub(path)
    tool = ToolStub(inp)

    result = _fusion_add_keyframe_to_input(comp, tool, "Center", inp, 12, [0.25, 0.75])

    assert "error" in result
    assert result["error"]["code"] == "FUSION_POINT_KEYFRAME_FAILED"


def test_scalar_keyframes_parse_indexed_time_readback():
    inp = IndexedScalarInputStub()
    tool = ToolStub(inp)

    result = _fusion_add_keyframe_to_input(None, tool, "Blend", inp, 84, 1.0)

    assert result["success"] is True
    assert _fusion_keyframe_times(inp) == [84]
    assert _fusion_read_keyframes(inp) == [{"time": 84, "value": 1.0}]
