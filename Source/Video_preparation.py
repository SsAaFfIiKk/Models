import cv2


class VideoPreparation:
    def __init__(self, path_to_video, chunk_size):
        self.cap = cv2.VideoCapture(path_to_video)
        self.frame_number = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.chunk_size = chunk_size
        self.number_of_chunks = self.frame_number / self.chunk_size
        self.step = self.chunk_size * self.fps
        self.chunks = []

    def make_chunks_borders(self):
        frame_start = 0
        for border in range(self.step, self.frame_number + 1, self.step):
            if frame_start == 0:
                self.chunks.append((frame_start, border))
                frame_start = border
            else:
                self.chunks.append((frame_start, border))
                frame_start = border
        self.chunks.append((frame_start, self.frame_number))

    def get_data(self):
        return self.cap, self.chunks, self.fps
