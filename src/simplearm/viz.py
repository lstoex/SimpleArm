from simplearm.robot import RobotInfo
from simplearm.kinematics import forward_kinematic, world_spheres_from_frames
import numpy as np
import plotly.graph_objects as go
import functools
from yaspin import yaspin

class RobotViewer:
    def __init__(
        self,
        q: np.ndarray,
        robot_info: RobotInfo,
        obstacles: np.ndarray | None = None,
    ):
        self.robot_info = robot_info
        self.dist_img = None
        self.obstacles = obstacles
        self.animate = True if q.ndim > 1 else False
        self.q = np.atleast_2d(q)

        self.frames = forward_kinematic(
            self.q, self.robot_info.linklengths
        )
        self.spheres = world_spheres_from_frames(
            self.frames, self.robot_info.spheres
        )
        self.current_q_idx = 0
        # register trace and shape getters
        self.trace_getters = [
            func
            for name, func in vars(self.__class__).items()
            if callable(func) and getattr(func, "_tracegetter", False)
        ]
        self.shapes_getters = [
            func
            for name, func in vars(self.__class__).items()
            if callable(func) and getattr(func, "_shapegetter", False)
        ]

        self.bgtrace_getters = [
            func
            for name, func in vars(self.__class__).items()
            if callable(func) and getattr(func, "_bgtracegetter", False)
        ]

        # create figure
        self.fig = go.Figure()
        max_robot_extent = np.sum(self.robot_info.linklengths) + 0.5
        self.fig.update_layout(
            xaxis=dict(
                # range=[self.world_info.limits[0, 0], self.world_info.limits[0, 1]]
                range=[-max_robot_extent, max_robot_extent]
            ),
            yaxis=dict(
                # range=[self.world_info.limits[1, 0], self.world_info.limits[1, 1]]
                range=[-max_robot_extent, max_robot_extent]
            ),
            yaxis_scaleanchor="x",
            yaxis_scaleratio=1,
            xaxis_scaleanchor="y",
            xaxis_scaleratio=1,
            showlegend=True,
            width=800,
            height=800,
            paper_bgcolor="white",
            plot_bgcolor="white",
        )

        # preconfigure spinner
        self.spinner = yaspin(text="Plotting robot...", timer=True)

    @staticmethod
    def tracegetter(func):
        """
        Decorator to mark a function as a trace getter. Trace getters are functions that return plotly traces.
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper._tracegetter = True # type: ignore
        return wrapper

    @staticmethod
    def shapegetter(func):
        """
        Decorator to mark a function as a shape getter. Shape getters are functions that return plotly shapes.
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper._shapegetter = True # type: ignore
        return wrapper

    @staticmethod
    def bgtracegetter(func):
        """
        Decorator to mark a function as a background trace getter. They return Plotly traces that are only plotted once
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper._bgtracegetter = True # type: ignore
        return wrapper

    def plot(self, duration=100):
        """
        Plot the robot.
        Args:
            duration (int): The duration of the animation in milliseconds. Only used if the robot is animated.
        """
        self.spinner.text = "Plotting robot..."
        self.spinner.start()
        try:
            if self.animate:
                frames = []
                for i in range(self.q.shape[0]):
                    frame = go.Frame(data=self.__get_frame_data(i), name=str(i))
                    frames.append(frame)
                fig = go.Figure(data=self.__get_frame_data(0), frames=frames)
                # copy layout from self.fig
                fig.layout = self.fig.layout
                self.fig = fig
                self.fig.frames = frames
                self.fig.update_layout(
                    updatemenus=[
                        {
                            "buttons": [
                                {
                                    "args": [
                                        None,
                                        {
                                            "frame": {
                                                "duration": duration,
                                                "redraw": True,
                                            },
                                            "fromcurrent": True,
                                        },
                                    ],
                                    "label": "Play",
                                    "method": "animate",
                                },
                                {
                                    "args": [
                                        [None],
                                        {
                                            "frame": {"duration": 0, "redraw": True},
                                            "mode": "immediate",
                                            "transition": {"duration": 0},
                                        },
                                    ],
                                    "label": "Pause",
                                    "method": "animate",
                                },
                            ],
                            "direction": "left",
                            "pad": {"r": 10, "t": 87},
                            "showactive": True,
                            "type": "buttons",
                            "x": 1.025,
                            "xanchor": "left",
                            "y": 0.4,
                            "yanchor": "top",
                        }
                    ],
                    sliders=[
                        {
                            "steps": [
                                {
                                    "args": [
                                        [str(i)],
                                        {
                                            "frame": {
                                                "duration": duration,
                                                "redraw": True,
                                            },
                                            "mode": "immediate",
                                            "transition": {"duration": 0},
                                        },
                                    ],
                                    "label": str(i),
                                    "method": "animate",
                                }
                                for i in range(len(frames))
                            ],
                            "transition": {"duration": 0},
                            "x": 0.5,
                            "len": 0.9,
                            "xanchor": "center",
                            "y": -0.0,
                            "yanchor": "top",
                            "currentvalue": {
                                "prefix": "Frame:",
                                "visible": True,
                                "xanchor": "right",
                            },
                            "pad": {"b": 10, "t": 50},
                        }
                    ],
                )
            else:
                self.fig.data = []  # clear all traces
                self.fig.layout.shapes = []  # clear all shapes
                for getter in self.trace_getters:
                    self.fig.add_traces(getter(self))
            for getter in self.shapes_getters:
                for shape in getter(self):  # shapes getter returns a list of shapes
                    self.fig.add_shape(shape)
            # self.fig.add_shape(
            #     type="rect",
            #     x0=self.world_info.limits[0, 0],
            #     y0=self.world_info.limits[1, 0],
            #     x1=self.world_info.limits[0, 1],
            #     y1=self.world_info.limits[1, 1],
            #     line=dict(width=3, color="gray"),
            # )
            for getter in self.bgtrace_getters:
                self.fig.add_traces(getter(self))
            self.fig.show()
        except Exception as e:
            self.spinner.fail()
            self.spinner.write("Failed to plot robot")
            raise e
        self.spinner.stop()

    def __get_frame_data(self, frame_idx):
        """
        Get the data for a single frame when animating the robot.
        Args:
            frame_idx (int): The index of the frame to get.
        Returns:
            List[go.Scatter]: A list of plotly traces.
        """
        self.current_q_idx = frame_idx
        data = []
        for getter in self.trace_getters:
            data.extend(getter(self))
        return data

    @tracegetter
    def __get_links(self):
        """
        Get the links of the robot.
        Returns:
            List[go.Scatter]: A list of plotly traces representing the links of the robot.
        """
        # q_base = self.q.base[self.current_q_idx]
        # q_fingers = [self.q.fingers[i][self.current_q_idx] for i in range(len(self.q.fingers))]
        # current_q = Configuration(q_base, q_fingers)
        # frames = self.get_single_entry(self.current_q_idx, self.frames)
        frames = self.frames[self.current_q_idx]
        links = []

        def plot_chain(name, frames):
            links = []
            for frame_id in range(len(frames) - 1):
                prev_origin = frames[frame_id][:-1, -1]
                link_origin = frames[frame_id + 1][:-1, -1]
                link = go.Scatter(
                    x=[prev_origin[0], link_origin[0]],
                    y=[prev_origin[1], link_origin[1]],
                    mode="lines + markers",
                    name=f"Link {name} {frame_id}",
                    line=dict(width=5),
                )
                links.append(link)
            return links

        links_base = plot_chain("", frames)
        links.extend(links_base)
        return links

    @tracegetter
    def __get_frames(self):
        """
        Get the frames of the robot.
        Returns:
            List[go.Scatter]: A list of plotly traces representing the frames of the robot.
        """
        frames = self.frames[self.current_q_idx]
        frame_plots = []
        init_legend = True
        for frame in frames:
            x_vec = frame[:-1, 0]
            y_vec = frame[:-1, 1]
            origin = frame[:-1, -1]
            # plot x axis
            x_axis = go.Scatter(
                x=[origin[0], origin[0] + x_vec[0] * 0.1],
                y=[origin[1], origin[1] + x_vec[1] * 0.1],
                mode="lines",
                name="X-Axis",
                line=dict(color="red"),
                showlegend=init_legend,
                legendgroup="frame",
                visible="legendonly",
            )
            # plot y axis
            y_axis = go.Scatter(
                x=[origin[0], origin[0] + y_vec[0] * 0.1],
                y=[origin[1], origin[1] + y_vec[1] * 0.1],
                mode="lines",
                name="Y-Axis",
                line=dict(color="blue"),
                showlegend=init_legend,
                legendgroup="frame",
                visible="legendonly",
            )
            init_legend = False
            frame_plots.append(x_axis)
            frame_plots.append(y_axis)
        return frame_plots

    def save_fig(self, file_path):
        """
        Save the current figure to a file.
        Args:
            file_path (str): The path to save the figure to.
        """
        self.fig.write_html(file_path, auto_play=False)
        self.spinner.write(f"Figure saved to {file_path}")

    @staticmethod
    def draw_filled_circle(center, radius, n_points=10, **kwargs):
        """
        Draw a filled circle.
        Args:
            center (Tuple[float, float]): The center of the circle.
            radius (float): The radius of the circle.
            n_points (int): The number of points to use for the circle approximation.
            **kwargs: Additional arguments to pass to the plotly Scatter objects.
        Returns:
            go.Scatter: A plotly trace representing the circle.
        """
        # we use a filled scatter plot to draw a circle
        # this is not ideal, but it works with animations
        circle = np.linspace(0, 2 * np.pi, num=n_points)
        x = center[0] + radius * np.cos(circle)
        y = center[1] + radius * np.sin(circle)
        return go.Scatter(
            x=x, y=y, mode="lines", fill="toself", line=dict(width=0), **kwargs
        )

    @tracegetter
    def __get_obstacles(self):
        if self.obstacles is None:
            return []
        else:
            traces = []
            for obs in self.obstacles:
                circle = self.draw_filled_circle(
                    center=obs[:2],
                    radius=obs[2],
                    fillcolor="gray",
                    opacity=0.5,
                    name="Obstacle",
                    legendgroup="obstacle",
                )
                traces.append(circle)
            return traces

    @tracegetter
    def __get_spheres(self):
        """
        Get the (approximated) spheres of the robot.
        Returns:
            List[go.Scatter]: A list of plotly traces representing the spheres of the robot.
        """
        spheres = []
        init_legend = True
        spheres_xy = np.stack([self.spheres.x, self.spheres.y], axis=-1)
        sphere_pos = spheres_xy[self.current_q_idx]
        sphere_r = self.robot_info.spheres.r
        for pos, rad in zip(sphere_pos, sphere_r):
            # we cannot use scatter as it would be resized with the plot. We need to use shapes which have to be passed as dicts
            circle = self.draw_filled_circle(
                pos,
                rad,
                fillcolor="orange",
                opacity=0.2,
                name="Collision Sphere",
                legendgroup="sphere",
                showlegend=init_legend,
            )
            init_legend = False
            spheres.append(circle)
        return spheres
