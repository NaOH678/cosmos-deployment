"""Offline benchmark server wrapper; existing production entrypoint remains unchanged."""
import os
from cosmos_framework.inference.robot_policy import omni_http
if os.environ.get('WAM_COMBINED_CANDIDATE') == '1':
    from first_frame_packet import build_first_frame_packet
    def make_packet(self, images, state):
        return build_first_frame_packet(self.config, images, state)
    omni_http.OmniWamAdapter._make_packet = make_packet
if __name__ == '__main__':
    omni_http.main()
