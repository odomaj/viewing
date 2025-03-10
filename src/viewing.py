from typing import Annotated
from dataclasses import dataclass
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt

Vertex = Annotated[np.typing.NDArray, "vector of size 3"]
Vertex_H = Annotated[np.typing.NDArray, "vector of size 4"]

Vertices = Annotated[np.typing.NDArray, "vector of size N x 3"]
Vertices_H = Annotated[np.typing.NDArray, "vector of size N x 4"]

Edge = Annotated[tuple[int, int], "tuple 2 ints"]


class ProjectionType(Enum):
    PERSPECTIVE = 1
    ORTHOGRAPHIC = 2


@dataclass
class Rectangle:
    height: np.float32
    width: np.float32
    depth: np.float32

    center: Vertex

    vertices: Vertices
    edges: list[Edge]


def normalize(vector: np.ndarray) -> np.ndarray:
    return vector / np.sqrt(np.dot(vector, vector))


def sqr_distance(a: Vertex, b: Vertex) -> np.float32:
    diff: Vertex = a - b
    return np.dot(diff, diff)


def close(a: Vertex, b: Vertices) -> int:
    close_i: int = 0
    close_d: np.float32 = sqr_distance(a, b[0])
    for i in range(1, len(b)):
        dis: np.float32 = sqr_distance(a, b[i])
        if dis < close_d:
            close_i = i
            close_d = dis
    return close_i


def far(a: Vertex, b: Vertices) -> int:
    far_i: int = 0
    far_d: np.float32 = sqr_distance(a, b[0])
    for i in range(1, len(b)):
        dis: np.float32 = sqr_distance(a, b[i])
        if dis > far_d:
            far_i = i
            far_d = dis
    return far_i


# Create a 3D cube
def centered_cube(side_length: np.float32 = 2) -> Rectangle:
    l = side_length / 2
    return Rectangle(
        height=side_length,
        width=side_length,
        depth=side_length,
        center=np.zeros(3, dtype=np.float32),
        vertices=np.array(
            [
                [-1 * l, -1 * l, -1 * l],
                [l, -1 * l, -1 * l],
                [l, l, -1 * l],
                [-1 * l, l, -1 * l],  # Back face
                [-1 * l, -1 * l, l],
                [l, -1 * l, l],
                [l, l, l],
                [-1 * l, l, l],  # Front face
            ],
            dtype=np.float32,
        ),
        edges=[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),  # Back face
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),  # Front face
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),  # Connecting edges
        ],
    )


def cube_at_point(point: Vertex, side_length: np.float32):
    p = point
    l = side_length
    return Rectangle(
        height=side_length,
        width=side_length,
        depth=side_length,
        center=np.array(
            [p[0] + l / 2, p[1] + 1 / 2, p[2] + 1 / 2], dtype=np.float32
        ),
        vertices=np.array(
            [
                p,
                [p[0] + l, p[1], p[2]],
                [p[0] + l, p[1] + l, p[2]],
                [p[0], p[1] + l, p[2]],  # close face
                [p[0], p[1], p[2] + l],
                [p[0] + l, p[1], p[2] + l],
                [p[0] + l, p[1] + l, p[2] + l],
                [p[0], p[1] + l, p[2] + l],  # far face
            ]
        ),
        edges=[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),  # close face
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),  # far face
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),  # Connecting edges
        ],
    )


# Camera Transformation
def camera_transform(
    vertices: Vertices,
    eye: Vertex,
    gaze: Vertex,
    up: Vertex,
) -> Vertices_H:
    """convert a set of vertices from the canonical basis of R^3 to an
    the homogenous coordinates of an orthonormal basis in R^3 centered at eye,
    generated from gaze and up"""

    # point away from objects
    w = -1 * normalize(gaze)
    # point right
    u = normalize(np.cross(up, w))
    # point up
    v = np.cross(w, u)

    # shape m x 4 (need transpose, and output transpose of transpose)
    homogenous_coords: Vertices_H = np.concatenate(
        (vertices, np.ones((vertices.shape[0], 1), dtype=np.float32)), axis=1
    )

    # shape 4 x 4
    translation: Vertex_H = np.array(
        [
            [1, 0, 0, -1 * eye[0]],
            [0, 1, 0, -1 * eye[1]],
            [0, 0, 1, -1 * eye[2]],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )

    # shape 4 x 4
    basis_change: Vertices_H = np.array(
        [
            [u[0], v[0], w[0], 0],
            [u[1], v[1], w[1], 0],
            [u[2], v[2], w[2], 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )

    # returning basis_change (B) * translation (T) * homogenous_coords_rows (H)
    #   (B * T * Ht)t
    #   = H * (B * T)t
    transformation = np.transpose(np.dot(basis_change, translation))
    return np.dot(homogenous_coords, transformation)


def project_perspective(
    vertices: Vertices_H,
    orthogonal_volume: Rectangle,
) -> Vertices_H:

    close_ = orthogonal_volume.vertices[
        close(np.zeros(3, dtype=np.float32), orthogonal_volume.vertices)
    ]
    far_ = orthogonal_volume.vertices[far(close_, orthogonal_volume.vertices)]

    # everything is scaled by a factor of z by this matrix
    transform: Vertices_H = np.array(
        [
            [close_[2], 0, 0, 0],
            [0, close_[2], 0, 0],
            [
                0,
                0,
                close_[2] + far_[2],
                -1 * close_[2] * far_[2],
            ],
            [0, 0, 1, 0],
        ],
        dtype=np.float32,
    )

    # need to scale the output vectors based on their z coordinate
    # not a linear transformation but only way to solve problem
    scaled_result: Vertices_H = np.dot(vertices, np.transpose(transform))
    for i in range(len(scaled_result)):
        scaled_result[i] = scaled_result[i] / scaled_result[i, -1]
    return scaled_result


def project_orthographic(
    vertices: Vertices_H,
    orthogonal_volume: Rectangle,
) -> Vertices_H:

    close_ = orthogonal_volume.vertices[
        close(np.zeros(3, dtype=np.float32), orthogonal_volume.vertices)
    ]
    far_ = orthogonal_volume.vertices[far(close_, orthogonal_volume.vertices)]

    transform: Vertices_H = np.array(
        [
            [
                2 / (far_[0] - close_[0]),
                0,
                0,
                (far_[0] + close_[0]) / (close_[0] - far_[0]),
            ],
            [
                0,
                2 / (far_[1] - close_[1]),
                0,
                (far_[1] + close_[1]) / (close_[1] - far_[1]),
            ],
            [
                0,
                0,
                2 / (close_[2] - far_[2]),
                (far_[2] + close_[2]) / (far_[2] - close_[2]),
            ],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )

    return np.dot(vertices, np.transpose(transform))


# Projection Transformation
def project_vertices(
    vertices: Vertices_H,
    orthogonal_volume: Rectangle,
    projection_type: ProjectionType = ProjectionType.PERSPECTIVE,
) -> Vertices_H | None:
    """
    Apply a projection transformation to 3D vertices.
    - perspective: applies a perspective projection.
    - orthographic: applies an orthographic projection.
    """
    if projection_type == ProjectionType.PERSPECTIVE:
        vertices = project_perspective(vertices, orthogonal_volume)
    elif projection_type != ProjectionType.ORTHOGRAPHIC:
        return None
    return project_orthographic(vertices, orthogonal_volume)


# Viewport Transformation
def viewport_transform(
    vertices: Vertices_H,
    target_width: np.float32,
    target_height: np.float32,
) -> Vertices:

    # blow up the image to the final size, and shift out of the
    # center (no negatives)
    viewport_matrix: Vertex_H = np.array(
        [
            [target_width / 2, 0, 0, (target_width - 1) / 2],
            [0, target_height / 2, 0, (target_height - 1) / 2],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )

    homogenous_coordinates = np.transpose(
        np.dot(viewport_matrix, np.transpose(vertices))
    )

    homogenous_coordinates = np.dot(vertices, np.transpose(viewport_matrix))

    return np.delete(homogenous_coordinates, -1, 1)


# Render the scene
def render_scene(vertices, edges, ax, **kwargs) -> None:
    for edge in edges:
        v1, v2 = vertices[edge[0]], vertices[edge[1]]
        ax.plot([v1[0], v2[0]], [v1[1], v2[1]], **kwargs)


# Main function
def main() -> int:
    # Scene setup
    cube = centered_cube()
    eye: Vertex = np.array([0.5, 0.5, -3])  # Camera at the origin
    gaze: Vertex = np.array([-1, -1, -5])  # Looking towards the cube
    up: Vertex = np.array([0, 1, 0])  # Up is along +Y-axis

    # Camera transformation
    transformed_vertices: Vertices_H = camera_transform(
        cube.vertices,
        eye,
        gaze,
        up,
    )

    orthogonal_volume: Rectangle = cube_at_point(
        np.array([-2.0, -2.0, 2.0], dtype=np.float32), 4.0
    )

    # Projection transformations
    perspective_vertices: Vertices_H = project_vertices(
        transformed_vertices,
        orthogonal_volume=orthogonal_volume,
        projection_type=ProjectionType.PERSPECTIVE,
    )

    orthographic_vertices: Vertices_H = project_vertices(
        transformed_vertices,
        orthogonal_volume=orthogonal_volume,
        projection_type=ProjectionType.ORTHOGRAPHIC,
    )

    # Viewport transformation
    viewport_width, viewport_height = 1920, 1080
    persp_2d: Vertices = viewport_transform(
        perspective_vertices, viewport_width, viewport_height
    )
    ortho_2d: Vertices = viewport_transform(
        orthographic_vertices, viewport_width, viewport_height
    )

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].set_title("Perspective Projection")
    axes[1].set_title("Orthographic Projection")

    render_scene(persp_2d, cube.edges, axes[0], color="blue", marker="o")
    render_scene(ortho_2d, cube.edges, axes[1], color="red", marker="o")

    for ax in axes:
        ax.set_xlim(0, viewport_width)
        ax.set_ylim(0, viewport_height)
        ax.set_aspect("equal")
        ax.invert_yaxis()

    plt.show()
    return 0


if __name__ == "__main__":
    exit(main())
