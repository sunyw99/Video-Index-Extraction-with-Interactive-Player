package view;

import javax.swing.*;
import java.awt.*;

public class MainFrame {

    public MainFrame(){
        JFrame frame = new JFrame("ShotDetector");

        frame.setBounds(new Rectangle(200, 200, 800, 600));

        CardLayout cardLayout = new CardLayout();
        JPanel cardPanel = new JPanel(cardLayout);

        VideoPlayer videoPlayer = new VideoPlayer();
        FileChooser fileChooser = new FileChooser(cardPanel, cardLayout, videoPlayer);

        cardPanel.add(fileChooser, "FC");
        cardPanel.add(videoPlayer, "VP");

        frame.getContentPane().add(cardPanel);

        frame.setVisible(true);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }
}