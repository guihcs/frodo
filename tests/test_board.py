import unittest
from board import Board

class BoardTestCase(unittest.TestCase):

    def test_move(self):

        nl = [(1, 1), (0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]

        for i in range(len(nl)):

            board = Board()
            board.board = [[1, 1, 1], [1, 2, 1], [1, 1, 1]]
            board.do_action((1, 1), (2, 2), i)
            target = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
            target[nl[i][0]][nl[i][1]] = 2
            self.assertEqual(board.board, target)

    def test_push(self):

            nl = [(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)]

            for i in range(len(nl)):

                board = Board()
                board.board = [[2, 1, 1], [1, 3, 1], [1, 1, 1]]
                board.do_action((0, 0), (1, 1), i + 9)
                target = [[2, 1, 1], [1, 1, 1], [1, 1, 1]]
                target[nl[i][0]][nl[i][1]] = 3
                self.assertEqual(board.board, target)

            board = Board()
            board.board = [[2, 3, 1], [1, 1, 1], [1, 1, 1]]
            board.do_action((0, 0), (0, 1), 10)
            target = [[2, 1, 1], [1, 3, 1], [1, 1, 1]]
            self.assertEqual(board.board, target)


    def test_throw_mw(self):

        board = Board()
        board.board = [[0, 3, 0], [0, 1, 0], [2, 1, 1]]
        board.do_action((2, 0), (1, 1),14)
        target = [[0, 3, 0], [0, 4, 0], [2, 1, 1]]
        self.assertEqual(board.board, target)


    def test_win(self):
        board = Board()
        board.board = [[0, 3, 0], [0, 4, 0], [2, 1, 1]]
        self.assertEqual(board.is_win(), True)


    def test_get_position(self):
        board = Board()
        board.board = [[0, 3, 0], [0, 4, 0], [2, 1, 1]]
        self.assertEqual(board.get_position(2), (2, 0))
        self.assertEqual(board.get_position(3), (0, 1))

if __name__ == '__main__':
    unittest.main()
