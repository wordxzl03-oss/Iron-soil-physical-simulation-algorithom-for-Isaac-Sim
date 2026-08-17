import ast
import inspect
import unittest

from isaac_bulk_pipeline.vehicle import KeyboardCommandState, ManualLoaderController
import isaac_bulk_pipeline.vehicle.manual_loader_controller as manual_module


class PhaseBManualControllerTests(unittest.TestCase):
    def test_press_release_and_opposing_key_contract(self):
        keyboard = KeyboardCommandState()
        keyboard.handle_event("W", True)
        self.assertEqual(keyboard.command().throttle, 1.0)
        keyboard.handle_event("S", True)
        self.assertEqual(keyboard.command().throttle, 0.0)
        keyboard.handle_event("W", False)
        self.assertEqual(keyboard.command().throttle, -1.0)
        keyboard.handle_event("S", False)
        self.assertEqual(keyboard.command().throttle, 0.0)

        for key in ("A", "I", "J", "SPACE"):
            keyboard.handle_event(key, True)
        command = keyboard.command()
        self.assertEqual(command.steering, 1.0)
        self.assertEqual(command.lift, 1.0)
        self.assertEqual(command.bucket_curl, 1.0)
        self.assertEqual(command.brake, 1.0)
        self.assertEqual(command.throttle, 0.0)

    def test_r_clears_only_command_state_and_escape_stops(self):
        controller = ManualLoaderController()
        controller.on_key_event("KeyboardInput.W", True)
        controller.on_key_event("R", True)
        self.assertEqual(controller.keyboard.command().throttle, 0.0)
        self.assertFalse(controller.stopped)
        controller.on_key_event("ESC", True)
        self.assertTrue(controller.stopped)
        self.assertEqual(controller.keyboard.pressed_keys, frozenset())

    def test_keyboard_callback_module_has_no_robot_or_pose_calls(self):
        tree = ast.parse(inspect.getsource(manual_module))
        called_attributes = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertTrue({"set_world_pose", "set_world_poses", "apply_action"}.isdisjoint(called_attributes))
        source = inspect.getsource(manual_module)
        self.assertNotIn("isaacsim", source)
        self.assertNotIn("carb.input", source)


if __name__ == "__main__":
    unittest.main()
