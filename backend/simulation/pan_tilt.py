class PanTilt:

    def __init__(self):

        self.azimuth = 0.0
        self.elevation = 0.0

        self.azimuth_velocity = 0.0
        self.elevation_velocity = 0.0

        self.max_azimuth_rate = 30.0
        self.max_elevation_rate = 20.0

        self.max_azimuth = 180.0
        self.min_azimuth = -180.0

        self.max_elevation = 90.0
        self.min_elevation = -30.0

    def update(
        self,
        azimuth_command,
        elevation_command,
        dt
    ):

        self.azimuth_velocity = max(
            -self.max_azimuth_rate,
            min(
                self.max_azimuth_rate,
                azimuth_command
            )
        )

        self.elevation_velocity = max(
            -self.max_elevation_rate,
            min(
                self.max_elevation_rate,
                elevation_command
            )
        )

        self.azimuth += (
            self.azimuth_velocity * dt
        )

        self.elevation += (
            self.elevation_velocity * dt
        )

        self.azimuth = max(
            self.min_azimuth,
            min(self.max_azimuth, self.azimuth)
        )

        self.elevation = max(
            self.min_elevation,
            min(self.max_elevation, self.elevation)
        )
