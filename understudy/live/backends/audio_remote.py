"""Placeholder for a later remote transcription model."""


class RemoteBackend:
    name = "remote"

    def __init__(self, model="remote", **_):
        self.model = model

    def run(self, payload):
        raise NotImplementedError("the remote transcription backend is not "
                                  "implemented yet")
