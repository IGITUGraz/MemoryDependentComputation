"""Simple Concentration-Game environment"""
import os
from typing import Tuple, Any, Optional, List, Union, Iterable

import gym
import gym.utils.seeding
import numpy as np
from numpy.random import Generator

from functions.utility_functions import to_one_hot


class Card(object):

    states = {'FACE_DOWN': 0, 'FACE_UP': 1}

    def __init__(self, image: Any, image_id: int) -> None:
        super().__init__()
        self.image = image
        self.image_id = image_id
        self.state = self.states['FACE_DOWN']

    def __eq__(self, other):
        if not isinstance(other, Card):
            return NotImplemented

        return self.image_id == other.image_id

    @property
    def is_face_down(self) -> bool:
        return self.state == self.states['FACE_DOWN']

    @property
    def is_face_up(self) -> bool:
        return self.state == self.states['FACE_UP']

    def flip(self) -> None:
        self.state = self.states['FACE_UP'] if self.state == self.states['FACE_DOWN'] else self.states['FACE_DOWN']

    def view(self) -> Any:
        if self.is_face_up:
            return self.image
        else:
            return np.zeros_like(self.image)


class Cell(object):

    states = {'EMPTY': 0, 'OCCUPIED': 1}

    def __init__(self, card: Optional[Card] = None) -> None:
        super().__init__()
        self.card = card
        self.state = self.states['EMPTY'] if card is None else self.states['OCCUPIED']

    @property
    def is_empty(self) -> bool:
        return self.state == self.states['EMPTY']

    @property
    def is_occupied(self) -> bool:
        return self.state == self.states['OCCUPIED']

    def remove_card(self) -> None:
        self.card = None
        self.state = self.states['EMPTY']


class Grid(object):

    cell_states = {'FACE_DOWN': 0, 'FACE_UP': 1, 'EMPTY': 2}

    def __init__(self, cells: List[Cell], card_shape: Union[int, Iterable, Tuple[int]]) -> None:
        super().__init__()
        self.cells = cells
        self.card_shape = card_shape
        self.grid = np.array(cells)

    def __call__(self, pos: int) -> Cell:
        return self.grid[pos]

    def remove_card_at(self, pos: int) -> None:
        self(pos).remove_card()

    def flip_card_at(self, pos: int) -> None:
        if not self(pos).is_empty:
            self(pos).card.flip()

    def view_card_at(self, pos: int) -> Any:
        if not self(pos).is_empty:
            return self(pos).card.view()
        else:
            return np.zeros(self.card_shape)

    def no_cards_left(self) -> bool:
        for cell in self.cells:
            if cell.is_occupied:
                return False
        return True

    def matching_pair_is_face_up(self) -> bool:
        cards = []
        for cell in self.cells:
            if not cell.is_empty and cell.card.is_face_up:
                cards.append(cell.card)
        if len(cards) == 2 and cards[0] == cards[1]:
            return True
        else:
            return False

    def tidy_up(self) -> None:
        cells = []
        for cell in self.cells:
            if not cell.is_empty and cell.card.is_face_up:
                cells.append(cell)
        if len(cells) == 2:
            if cells[0].card == cells[1].card:
                cells[0].remove_card()
                cells[1].remove_card()
            else:
                cells[0].card.flip()
                cells[1].card.flip()

    def get_cell_states(self) -> np.ndarray:
        states = []
        for cell in self.cells:
            cell_state = 'EMPTY' if cell.is_empty else 'FACE_UP' if cell.card.is_face_up else 'FACE_DOWN'
            states += to_one_hot(self.cell_states[cell_state], len(self.cell_states))

        return np.array(states)

    def empty_cells(self) -> np.ndarray:
        empty = []
        for cell in self.cells:
            if cell.is_empty:
                empty.append(True)
            else:
                empty.append(False)

        return np.array(empty)

    def face_up_cells(self) -> np.array:
        face_up = []
        for cell in self.cells:
            if not cell.is_empty and cell.card.is_face_up:
                face_up.append(True)
            else:
                face_up.append(False)

        return np.array(face_up)

    def display_grid(self) -> np.ndarray:
        grid = np.zeros_like(self.grid, dtype='<U3')
        for i, cell in enumerate(self.cells):
            if cell.is_empty:
                label = ' x '
            elif cell.card.is_face_up:
                label = '+' + str(cell.card.image_id) + '+'
            else:
                label = '-' + str(cell.card.image_id) + '-'
            grid[i] = label

        return grid


class ConcentrationObservationSpace(gym.Space):

    def __init__(self, num_cells: int, num_states: int, card_shape: int,
                 dtype: Optional[np.dtype] = np.float32) -> None:
        self.num_cells = num_cells
        self.num_states = num_states
        self.card_shape = card_shape
        self.dtype = dtype

        self.shape = (num_cells * num_states + num_cells + card_shape,)

        super().__init__(self.shape, self.dtype)

    def sample(self):
        obs = []
        for i in range(self.num_cells+1):  # Cell states +1 for position
            one_hot_vector = np.zeros(self.num_states)
            one_hot_vector[np.random.randint(self.num_states)] = 1
            obs.append(one_hot_vector)
        obs.append(np.random.random(self.card_shape))

        return np.hstack(obs)

    def contains(self, x):
        if isinstance(x, list):
            x = np.array(x)

        if not x.shape == self.shape:
            return False

        for i in range(self.num_cells):
            start = i * self.num_states
            stop = start + self.num_states
            if np.count_nonzero(x[start:stop]) != 1:
                return False

        if not (np.all(x[-self.card_shape:] >= 0.0) and np.all(x[-self.card_shape:] < 1.0)):
            return False

        return True


class Concentration(gym.Env):

    metadata = {'render.modes': ['ansi']}
    reward_range = (-float('inf'), float('inf'))
    spec = None

    def __init__(self, num_cells: int, card_shape: int, match_reward: float, flip_penalty: float,
                 resample_cards: bool) -> None:
        super().__init__()
        self.num_cells = num_cells
        self.card_shape = card_shape
        self.match_reward = match_reward
        self.flip_penalty = flip_penalty
        self.resample_cards = resample_cards

        self.action_space = gym.spaces.Discrete(num_cells)
        self.observation_space = ConcentrationObservationSpace(num_cells, len(Grid.cell_states), card_shape)

        self.rng = None
        self.grid = None
        self.num_steps = None
        self.episode_reward = None

        self.seed()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        position = action.item()
        assert self.action_space.contains(position), "%r (%s) invalid" % (position, type(position))

        reward = self._flip_card_at(position)
        if self.grid.matching_pair_is_face_up():
            reward += self.match_reward

        image = self.grid.view_card_at(position)

        self.grid.tidy_up()  # If two cards are face up, remove them if they match, otherwise turn them face down

        cell_states = self.grid.get_cell_states()
        position = to_one_hot(position, self.num_cells)

        self.num_steps += 1
        self.episode_reward += reward

        done = True if self.grid.no_cards_left() else False

        info = {'grid': self.grid.display_grid(),
                'num_steps': self.num_steps,
                'episode_reward': self.episode_reward}

        return np.hstack([cell_states, position, image]), reward, done, info

    def reset(self) -> np.ndarray:
        self.num_steps = 0
        self.episode_reward = 0

        images = self._get_images()
        self.grid = self._create_grid(images)

        # Create initial observation
        cell_states = self.grid.get_cell_states()
        position = to_one_hot(0, self.num_cells)
        image = np.zeros(self.card_shape)

        return np.hstack([cell_states, position, image])

    def render(self, mode: Optional[str] = 'ansi') -> np.ndarray:
        return self.grid.display_grid()

    def close(self) -> None:
        pass

    def seed(self, seed=None) -> List[int]:
        self.rng, seed = gym.utils.seeding.np_random(seed)

        return [seed]

    def _flip_card_at(self, position: int) -> float:
        self.grid.flip_card_at(position)

        return self.flip_penalty

    def _get_images(self) -> np.array:
        if self.resample_cards:
            return self.rng.random(size=(self.num_cells // 2, self.card_shape))
        else:
            file_path = os.path.join('./data', 'ConcentrationDataset', 'images-{0}.npy'.format(self.num_cells))
            if os.path.isfile(file_path):
                images = np.load(file_path, allow_pickle=True)
            else:
                images = self.rng.random(size=(self.num_cells // 2, self.card_shape))
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                np.save(file_path, images)

            return images

    def _create_grid(self, images: np.ndarray) -> Grid:
        cells = []
        for i, image in enumerate(images):
            cells.append(Cell(Card(image, image_id=i)))
            cells.append(Cell(Card(image, image_id=i)))
        self.rng.shuffle(cells)
        if len(cells) != self.num_cells:
            cells.append(Cell(card=None))  # Add empty cell if `num_cells` is odd

        return Grid(cells, self.card_shape)
