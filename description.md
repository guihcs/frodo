

# Environment Description

The environment consist of a 4x4 grid world. The rows from 0 to 3 and the columns also from 0 to 3.
The cells (0, 1), (0, 3), (1, 1), (1, 3) are blocked and the agents cannot move to these cells.
Considering the empty space as '.' and the blocked cells as '%' the environment looks like this:

|   | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 | . | % | . | % |
| 1 | . | % | . | % |
| 2 | . | . | . | . |
| 3 | . | . | . | . |

## Todd

In the environment there is also a random walker Todd 'T' that moves in the grid.
It can only move orthogonally (up, down, left, right) and can only access the cells (1, 2), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3) marked with '-'.
The walker moves randomly with equal probability to any direction with a delay of 2 seconds between each move.

|   | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 | . | % | . | % |
| 1 | . | % | - | % |
| 2 | . | - | T | - |
| 3 | . | - | . | - |

## Agent

The Player 'P' agent competes with another agent Enemy 'E'. The player and the enemy have the same action space. The actions are:
- Do nothing.
- Move orthogonally (up, down, left, right) and move diagonally (up-left, up-right, down-left, down-right).
- Push the enemy into any cell in the grid and also do double push.
- Throw magic walls in all empty cells except the cells (0, 0) and (0, 2).

## Movement

The player and the enemy can move orthogonally (up, down, left, right) and diagonally (up-left, up-right, down-left, down-right).
The player and the enemy can move to any cell in the grid except the blocked cells (0, 1), (0, 3), (1, 1), (1, 3) and the cells occupied by the walker Todd 'T'.
Also, the player, the enemy, and Todd cannot move to cells occupied by the magic walls.
The diagonal movement is also a bit slower than the orthogonal movement.

## Push

The player and the enemy have two types of push actions. The close push and the far push.
The close push happens when the player is adjacent to the enemy or the Todd and releases it to any cell in the grid.
If after 1 second the enemy or Todd is adjacent to the released cell it will be moved to the cell.
The far push happens when the player is not adjacent to the enemy or the Todd and releases it to any cell in the grid.
That causes the push counter to be increased to 1.6 and starts counting as soon as the action is executed. After that
the player will start to move in the direction of the grabbed entity by the shortest path. When the player reaches the entity the player stops moving.
If the timer reaches the 1.6 seconds and the enemy is near the released cell the enemy will be moved to the cell.
If the player is at 2 cells of distance it can perform also the close push. For example if the player is at (1, 0) and the enemy is at
(3, 1) and the player releases the enemy at (3, 3) the player will move to (2, 0) by the far push and in the next step it can perform the close
push to (3, 2) that will happen in 1 second while the far push will happen in 1.6 seconds. Since the close push will be executed first
the enemy will be moved to (3, 2) and the far push will still happen resulting in the enemy moving two cells. The push can be executed to any cell
even if it is occupied by the magic walls or is a blocked cell. The verification if the push is valid is done after the push delay.

## Magic Walls

The player can throw magic walls in all empty cells except the cells (0, 0) and (0, 2). The magic walls are represented by '#' and creates a barrier in the grid.
The magic walls last 20 seconds and the player have a cooldown of 2 seconds to throw another magic wall. The magic walls blocks the vision of throwing another magic wall.
If after drawing a line between the cell that the player wants to throw the magic wall and the cell that the player is the line crosses a magic wall the magic wall will not be thrown.
No one can move to cells occupied by the magic walls. However, the magic wall don't block the execution of pushes.

## Goal

The goal of the player is to trap the enemy in the cells (0, 0) or (0, 2) with magic walls in the cells (1, 0) and (1, 2) respectively.
If the enemy is trapped the player wins the game. If the player is trapped by the enemy the player loses the game.
If the player throws the magic wall in the cell (1, 0) or (1, 2) without the enemy being in the cell above the player loses the game.

## Rewards

- The player receives a reward of 20 when the enemy is trapped.
- The player receives a reward of -20 when the player is trapped.
- The player receives a reward of -20 when the player throws the magic wall in the cells (1, 0) or (1, 2) without the enemy being in the cells above.
- All actions have a cost of -1. Diagonal movements have a cost of -1.5 since they are slower and throws magic walls have a cost of -4.

