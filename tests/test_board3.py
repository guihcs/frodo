import unittest
from board3 import Board3

class Board3TestCase(unittest.TestCase):

    def test_board_creation(self):

        for _ in range(1000):
            b = Board3()
            self.assertTrue('E' in str(b) and 'P' in str(b) and 'T' in str(b))


    def test_move(self):
        b = Board3()
        b.set_player(0, 0)
        b.set_enemy(3, 3)
        b.player_walk(1, 0)
        self.assertEqual(b.get_player_position(), (1, 0))

    def test_move_fail(self):
        b = Board3()
        b.set_player(0, 0)
        b.set_enemy(3, 3)
        b.player_walk(1, 1)
        self.assertEqual(b.get_player_position(), (0, 0))

        b.player_walk(-1, -1)
        self.assertEqual(b.get_player_position(), (0, 0))

    def test_enemy_move_fail(self):
        b = Board3()
        b.set_enemy(0, 0)
        b.set_player(3, 3)
        b.set_todd(1, 2)
        b.player_throw_mw(1, 0)
        b.swap_enemy()
        b.push_enemy(3, 2)
        b.swap_enemy()
        b.step(1000)
        self.assertEqual(b.get_enemy_position(), (0, 0))

    def test_enemy_move_not_clearing_push(self):
        pass

    def test_push_enemy(self):
        b = Board3()
        b.set_player(2, 0)
        b.set_enemy(1, 0)
        b.push_enemy(0, 0)
        b.step(1000)
        self.assertEqual(b.get_enemy_position(), (0, 0))

    def test_push_enemy_fail(self):
        b = Board3()
        b.set_player(2, 2)
        b.set_enemy(2, 3)
        b.set_todd(1, 2)
        b.push_enemy(1, 2)
        b.step(1000)
        self.assertEqual(b.get_enemy_position(), (2, 3))

    def test_push_cancel(self):
        b = Board3()
        b.set_player(2, 3)
        b.set_enemy(2, 1)
        b.set_todd(3, 3)
        b.push_enemy(1, 0)
        b.push_enemy(3, 1)
        b.step(1600)
        self.assertEqual((3, 1), b.get_enemy_position())
        b.push_enemy(3, 2)
        b.push_enemy(2, 1)
        b.step(1000)
        self.assertEqual((2, 1), b.get_enemy_position())

    def test_long_push_fail_after_timer(self):
        board = Board3(walk_time=500)
        board.set_player(3, 2)
        board.set_enemy(1, 2)
        board.set_todd(3, 3)

        board.step(1000)
        board.push_enemy(0, 2)
        board.step(1000)
        self.assertEqual((1, 2), board.get_enemy_position())

    def test_long_push_reset_walk(self):
        board = Board3(walk_time=500)
        board.set_player(3, 2)
        board.set_enemy(1, 2)
        board.set_todd(3, 3)
        board.push_enemy(0, 2)
        board.step(1000)
        self.assertEqual((1, 2), board.get_enemy_position())
        board.player_walk(0, -1)
        board.step(1000)
        self.assertEqual((1, 2), board.get_enemy_position())


    def test_stop_chase_when_touch_long_push(self):
        board = Board3()

        board.set_enemy(1, 0)
        board.set_player(3, 1)
        board.set_todd(3, 3)
        board.player_throw_mw(2, 0)
        board.step(2000)

        board.push_enemy(0, 0)
        board.step(200)
        board.swap_enemy()
        board.player_walk(0, 0)
        board.step(200)

        self.assertEqual(board.get_enemy_position(), (2, 1))

    def test_long_push(self):
        b = Board3()
        b.set_player(3, 1)
        b.set_enemy(1, 0)
        b.set_todd(3, 3)

        b.player_throw_mw(2, 0)
        b.push_enemy(0, 0)
        b.step(200)

        self.assertEqual(b.get_player_position(), (2, 1))

    def test_shortest_path(self):
        b = Board3()
        b.set_player(3, 1)
        b.set_enemy(1, 0)
        b.set_todd(3, 3)

        b.player_throw_mw(2, 0)
        b.push_enemy(0, 0)

        path = b.shortest_path(3, 1, 1, 0)

        self.assertEqual(path, [(2, 1)])

    def test_double_push(self):
        b = Board3()
        b.set_todd(2, 3)
        b.set_player(2, 2)
        b.set_enemy(2, 0)
        b.push_enemy(0, 0)
        b.step(200)
        b.push_enemy(1, 0)
        b.step(1400)
        self.assertEqual(b.get_enemy_position(), (0, 0))

    def test_long_push_fail(self):
        b = Board3()
        b.set_player(0, 0)
        b.set_enemy(3, 3)
        b.set_todd(1, 2)
        b.player_throw_mw(1, 0)
        b.push_enemy(3, 2)
        b.step(1000)
        self.assertEqual(b.get_player_position(), (0, 0))

    def test_push_todd(self):
        b = Board3()
        b.set_enemy(0, 0)
        b.set_todd(2, 3)
        b.set_player(2, 2)
        b.push_todd(1, 2)
        b.step(1000)
        self.assertEqual(b.get_todd_position(), (1, 2))

    def test_double_push_todd(self):
        b = Board3()
        b.set_enemy(0, 0)
        b.set_todd(3, 3)
        b.set_player(2, 1)
        b.push_todd(1, 2)
        b.step(200)
        b.push_todd(2, 3)
        b.step(1400)
        self.assertEqual(b.get_todd_position(), (1, 2))


    def test_push_todd_fail(self):
        b = Board3()
        b.set_enemy(0, 0)
        b.set_todd(1, 2)
        b.set_player(2, 2)
        b.push_todd(0, 2)
        b.step(1000)
        self.assertEqual(b.get_todd_position(), (1, 2))

    def test_todd_wandering(self):
        b = Board3()
        b.set_enemy(0, 0)
        b.set_todd(2, 2)
        b.set_player(0, 2)
        b.step(3000)
        self.assertFalse(b.get_todd_position() == (2, 2))

    def test_todd_wandering_fail(self):
        b = Board3()
        b.set_enemy(0, 0)
        b.set_todd(1, 2)
        b.set_player(1, 0)
        b.player_throw_mw(2, 2)
        b.step(3000)
        self.assertTrue(b.get_todd_position() == (1, 2))

    def test_throw_mw_vertical(self):
        b = Board3()
        b.set_player(0, 0)
        b.set_enemy(0, 2)
        b.player_throw_mw(2, 0)
        self.assertFalse(b.can_move(2, 0))

    def test_throw_mw_horizontal(self):
        b = Board3()
        b.set_player(3, 0)
        b.set_enemy(0, 0)
        b.player_throw_mw(3, 2)
        self.assertFalse(b.can_move(3, 2))

    def test_throw_mw_fail(self):

        b = Board3()
        b.set_player(0, 0)
        b.set_enemy(0, 2)
        b.player_throw_mw(0, 2)
        b.player_throw_mw(3, 0)
        self.assertFalse((3, 0) in b.mw)
        b.step(1000)
        b.swap_enemy()
        b.player_throw_mw(1, 2)
        b.swap_enemy()
        b.step(1000)
        b.player_throw_mw(1, 2)
        b.step(1000)
        b.player_throw_mw(2, 0)
        b.step(5000)
        b.player_throw_mw(3, 0)
        self.assertTrue(b.can_move(3, 0))




    def test_mw_timer(self):
        b = Board3()
        b.set_player(0, 0)
        b.set_enemy(0, 2)
        b.player_throw_mw(3, 0)
        self.assertFalse(b.can_move(3, 0))
        b.step(10000)
        self.assertFalse(b.can_move(3, 0))
        b.step(10000)
        self.assertTrue(b.can_move(3, 0))

    def test_copy(self):
        b = Board3()
        b.set_player(0, 0)
        b.set_enemy(1, 2)
        b.set_todd(3, 3)
        b.player_throw_mw(3, 0)
        b.push_enemy(2, 3)
        c = b.copy()
        b.step(2500)
        c.step(2500)
        self.assertEqual(b.get_player_position(), c.get_player_position())
        self.assertEqual(b.get_enemy_position(), c.get_enemy_position())
        self.assertEqual(b.get_todd_position(), c.get_todd_position())
        self.assertEqual(b.mw, c.mw)

    def test_swap(self):
        b = Board3()
        b.set_player(0, 0)
        b.set_enemy(0, 2)
        b.swap_enemy()
        self.assertEqual(b.get_player_position(), (0, 2))
        self.assertEqual(b.get_enemy_position(), (0, 0))

    def test_enemy_push(self):
        b = Board3()
        b.set_todd(3, 3)
        b.set_player(1, 0)
        b.set_enemy(2, 0)
        b.swap_enemy()
        b.push_enemy(0, 0)
        b.swap_enemy()
        b.step(1000)
        self.assertEqual(b.get_player_position(), (0, 0))


    def test_enemy_double_push(self):
        b = Board3()
        b.set_todd(3, 3)
        b.set_player(2, 0)
        b.set_enemy(2, 2)
        b.swap_enemy()
        b.push_enemy(0, 0)
        b.step(200)
        b.push_enemy(1, 0)
        b.swap_enemy()
        b.step(1400)

        self.assertEqual(b.get_player_position(), (0, 0))

    def test_enemy_push_todd(self):
        b = Board3()
        b.set_player(0, 0)
        b.set_enemy(2, 1)
        b.set_todd(2, 2)
        b.swap_enemy()
        b.push_todd(1, 2)
        b.swap_enemy()
        b.step(1000)
        self.assertEqual(b.get_todd_position(), (1, 2))

    def test_enemy_double_push_todd(self):
        b = Board3()
        b.set_player(0, 0)
        b.set_enemy(2, 1)
        b.set_todd(3, 3)
        b.swap_enemy()
        b.push_todd(1, 2)
        b.step(200)
        b.push_todd(2, 3)
        b.swap_enemy()
        b.step(1400)
        self.assertEqual(b.get_todd_position(), (1, 2))




    def test_hash(self):
        b1 = Board3()
        b1.players_positions = [(2, 1), (0, 0), (3, 3)]
        b2 = Board3()
        b2.players_positions = [(2, 1), (0, 0), (3, 3)]
        b2.current_time = 5000
        b2.ev.events[0] = (8000, 0)
        b1.player_throw_mw(2, 1)
        b2.player_throw_mw(2, 1)

        self.assertEqual(hash(b1), hash(b2))
        self.assertEqual(b1, b2)

        b1 = Board3()
        b1.players_positions = [(2, 1), (1, 0), (3, 3)]
        b2 = Board3()
        b2.players_positions = [(2, 1), (1, 0), (3, 3)]
        b2.current_time = 5000
        b2.ev.events[0] = (8000, 0)
        b1.push_enemy(0, 0)
        b2.push_enemy(0, 0)

        self.assertEqual(hash(b1), hash(b2))
        self.assertEqual(b1, b2)
