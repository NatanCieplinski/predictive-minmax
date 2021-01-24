import chess

from engine.values import Values

class Heuristics:
    @staticmethod
    def material_heuristic(board):
        material_advantage = 0

        for piece_type in chess.PIECE_TYPES:
            white_pieces = board.pieces(piece_type, chess.WHITE)
            black_pieces = board.pieces(piece_type, chess.BLACK)
            material_advantage += Values.MATERIAL[piece_type - 1] * len(
                white_pieces) - Values.MATERIAL[piece_type - 1] * len(black_pieces)

        return material_advantage

    @staticmethod
    def piece_square_table_heuristic(board):
        positional_advantage = 0

        for piece_type in chess.PIECE_TYPES:

            for square in board.pieces(piece_type, chess.WHITE):
                position = 7 * (7 - int((square - 1) / 8)) + (square - 1) % 8
                positional_advantage += Values.PIECES_SQUARE_TABLES[piece_type - 1][position]

            for square in chess.flip_vertical(board.pieces(piece_type, chess.BLACK)):
                position = 7 * (7 - int((square - 1) / 8)) + (square - 1) % 8
                positional_advantage -= Values.PIECES_SQUARE_TABLES[piece_type - 1][position]

        return positional_advantage

    @staticmethod
    def attack_heuristic(board):
        attack_advantage = 0

        for piece_type in chess.PIECE_TYPES:
            for piece in board.pieces(piece_type, chess.WHITE):
                attack_advantage += len(chess.SquareSet(board.attacks_mask(piece)))

        for piece_type in chess.PIECE_TYPES:
            for piece in board.pieces(piece_type, chess.BLACK):
                attack_advantage -= len(chess.SquareSet(board.attacks_mask(piece)))

        return attack_advantage

    @staticmethod
    def evaluate_board(board):
        """Calculate the total heuristic"""
        return Heuristics.material_heuristic(board) + 0.5 * Heuristics.piece_square_table_heuristic(board) + 0.3 * Heuristics.attack_heuristic(board)