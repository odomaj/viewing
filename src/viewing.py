from typing import Annotated, Literal
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


def normalize(vector: np.ndarray) -> np.ndarray:
    return vector / np.sqrt(np.dot(vector, vector))


# Create a 3D cube
def create_cube() -> tuple[Vertices, list[Edge]]:
    vertices: Vertices = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],  # Back face
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],  # Front face
        ]
    )
    edges: list[Edge] = [
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
    ]
    return vertices, edges


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
    w = -1 * normalize(gaze)
    u = normalize(np.cross(up, w))
    v = np.cross(w, u)

    transformation: Vertices_H = np.dot(
        np.array(
            [
                np.append(u, 0),
                np.append(v, 0),
                np.append(w, 0),
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [1, 0, 0, -1 * eye[0]],
                [0, 1, 0, -1 * eye[1]],
                [0, 0, 1, -1 * eye[2]],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        ),
    )

    homogenous_coords: Vertices_H = np.concatenate(
        (vertices, np.ones((vertices.shape[0], 1), dtype=np.float32)), axis=1
    )

    return np.dot(homogenous_coords, np.transpose(transformation))


def project_perspective(
    vertices: Vertices_H,
    near: np.int32,
    far: np.int32,
    fov: np.float32,
    aspect: np.float32,
) -> Vertices_H:
    near_vector: Vertex = vertices[near][:-1]
    far_vector: Vertex = vertices[far][:-1]

    transform: Vertices_H = np.dot(
        np.array(
            [
                [
                    2 / (far_vector[0] - near_vector[0]),
                    0,
                    0,
                    (far_vector[0] + near_vector[0])
                    / (near_vector[0] - far_vector[0]),
                ],
                [
                    0,
                    2 / (far_vector[1] - near_vector[1]),
                    0,
                    (far_vector[1] + near_vector[1])
                    / (near_vector[1] - far_vector[1]),
                ],
                [
                    0,
                    0,
                    2 / (near_vector[2] - far_vector[2]),
                    (near_vector[2] + far_vector[2])
                    / (far_vector[2] - near_vector[2]),
                ],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [near_vector[2], 0, 0, 0],
                [0, near_vector[2], 0, 0],
                [
                    0,
                    0,
                    near_vector[2] + far_vector[2],
                    -1 * near_vector[2] * far_vector[2],
                ],
                [0, 0, 1, 0],
            ],
            dtype=np.float32,
        ),
    )

    return np.dot(vertices, np.transpose(transform))


def project_orthographic(
    vertices: Vertices_H,
    near: np.int32,
    far: np.int32,
) -> Vertices_H:
    near_vector: Vertex = vertices[near][:-1]
    far_vector: Vertex = vertices[far][:-1]

    transform: Vertices_H = np.array(
        [
            [
                2 / (far_vector[0] - near_vector[0]),
                0,
                0,
                (far_vector[0] + near_vector[0])
                / (near_vector[0] - far_vector[0]),
            ],
            [
                0,
                2 / (far_vector[1] - near_vector[1]),
                0,
                (far_vector[1] + near_vector[1])
                / (near_vector[1] - far_vector[1]),
            ],
            [
                0,
                0,
                2 / (near_vector[2] - far_vector[2]),
                (near_vector[2] + far_vector[2])
                / (far_vector[2] - near_vector[2]),
            ],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )

    return np.dot(vertices, np.transpose(transform))


# Projection Transformation
def project_vertices(
    vertices: Vertices_H,
    projection_type: ProjectionType = ProjectionType.PERSPECTIVE,
    near: np.int32 = 1,
    far: np.int32 = 10,
    fov: np.float32 = np.pi / 4,
    aspect: np.float32 = 1.0,
) -> Vertices_H | None:
    """
    Apply a projection transformation to 3D vertices.
    - perspective: applies a perspective projection.
    - orthographic: applies an orthographic projection.
    """
    if projection_type == ProjectionType.PERSPECTIVE:
        return project_perspective(vertices, near, far, fov, aspect)
    elif projection_type == ProjectionType.ORTHOGRAPHIC:
        return project_orthographic(vertices, near, far)
    return None


# Viewport Transformation
def viewport_transform(
    vertices: Vertices_H,
    width: np.float32,
    height: np.float32,
) -> Vertices:
    viewport_matrix: Vertex_H = np.array(
        [
            [width / 2, 0, 0, (width + 1) / 2],
            [0, height / 2, 0, (height + 1) / 2],
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
    vertices, edges = create_cube()
    eye: Vertex = np.array([0.5, 0.5, -3])  # Camera at the origin
    gaze: Vertex = np.array([-1, -1, -5])  # Looking towards the cube
    up: Vertex = np.array([0, 1, 0])  # Up is along +Y-axis

    # Camera transformation
    transformed_vertices: Vertices_H = camera_transform(
        vertices,
        eye,
        gaze,
        up,
    )

    # Projection transformations
    perspective_vertices: Vertices_H = project_vertices(
        transformed_vertices,
        ProjectionType.PERSPECTIVE,
        near=1,
        # far=10,
        far=7,
        fov=np.pi / 4,
        aspect=800 / 600,
    )

    orthographic_vertices: Vertices_H = project_vertices(
        transformed_vertices,
        ProjectionType.ORTHOGRAPHIC,
        near=1,
        # far=10,
        far=7,
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

    render_scene(persp_2d, edges, axes[0], color="blue", marker="o")
    render_scene(ortho_2d, edges, axes[1], color="red", marker="o")

    for ax in axes:
        ax.set_xlim(0, viewport_width)
        ax.set_ylim(0, viewport_height)
        ax.set_aspect("equal")
        ax.invert_yaxis()

    plt.show()
    return 0


if __name__ == "__main__":
    exit(main())
