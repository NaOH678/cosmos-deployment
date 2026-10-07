#include <memory>

#include "rclcpp/rclcpp.hpp"

#include "ManusDataPublisher.hpp"

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ManusDataPublisher>();
    rclcpp::spin(node);

    node.reset();

    if (rclcpp::ok()) {
        rclcpp::shutdown();
    }
    return 0;
}
