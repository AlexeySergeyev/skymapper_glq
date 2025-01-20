from __future__ import annotations
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


class Tstars:
    def __init__(self, num_stars: int) -> None:
        self.x = np.zeros(num_stars)
        self.y = np.zeros(num_stars)
        self.m = np.ones(num_stars)
        self.nuber_of_stars = num_stars

    def __repr__(self) -> str:
        np.set_printoptions(precision=2)
        return f"Stars number = {self.nuber_of_stars}\nx={self.x[:5]}...{self.x[-5:]}\ny={self.y[:5]}...{self.y[-5:]}\nm={self.m[:5]}...{self.m[-5:]}"

    def render_stars(self):
        plt.figure(figsize=(10, 10))
        plt.scatter(self.x, self.y, s=self.m * 10)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()


class Raytracer:
    def __init__(self, kappa: float, gamma: float, n: int, Re: float) -> None:
        self.kappa = kappa
        self.gamma = gamma
        self.Re = Re
        self.nx = n
        self.ny = n

        self.magnification = self.compute_magnification()
        self.nuber_of_stars = self.compute_number_of_stars()
        self.stars = Tstars(self.nuber_of_stars)


    def compute_magnification(self) -> float:
        magnification = 1.0 / np.abs((1.0 - self.kappa) ** 2 - self.gamma**2)
        print(f"Magnification = {magnification:.2f}")
        return magnification

    def compute_number_of_stars(self) -> int:
        number_of_stars = int(
            self.kappa * self.magnification * np.pi * (self.nx / 2) ** 2 / self.Re ** 2
        )

        print(f"Number of stars = {number_of_stars}")
        return number_of_stars

    def generate_stars_map(self) -> Tstars:
        """
        Generate a set of stars with random positions and Einstein radii.

        Parameters:
        stars (Tstars): An object containing the positions and Einstein radii of the stars.

        Returns:
        numpy.ndarray: A 2D array containing the positions and Einstein radii of the stars.
        """

        # Linear area in source plane
        self.lx = (1.0 - self.kappa + self.gamma) * self.magnification * self.nx
        self.ly = (1.0 - self.kappa - self.gamma) * self.magnification * self.ny

        rad = max(self.lx, self.ly) / 2.0
        print(f"Radius = {rad:.2f}")

        # Generate random positions for the stars
        i = 0
        np.random.seed(42)
        while i < self.nuber_of_stars:
            x = np.random.uniform(-rad, rad)
            y = np.random.uniform(-rad, rad)
            if np.sqrt(x**2 + y**2) < rad:
                self.stars.x[i] = x
                self.stars.y[i] = y
                i += 1

        print(self.stars)

        return 0


if __name__ == "__main__":
    # num_stars = 100
    # stars = Tstars(num_stars)
    rt = Raytracer(kappa=0.2, gamma=0.0, n=1000, Re=20.0)
    rt.generate_stars_map()
    rt.stars.render_stars()
