package view;

import tool.ProcessTool;

import javax.imageio.ImageIO;
import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.AreaAveragingScaleFilter;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.jar.JarEntry;

public class FileChooser extends JPanel {

    private String videoUrl;
    private String rgbUrl;

    public FileChooser(JPanel cardPanel, CardLayout cardLayout, VideoPlayer videoPlayer) {
        setLayout(new FlowLayout(FlowLayout.LEFT));

        JPanel buttonBars = new JPanel();

        JButton mp4Button = new JButton("Choose MP4");
        JButton rgbButton = new JButton("Choose RGB");
        JButton wavButton = new JButton("Choose WAV");
        JButton processButton = new JButton("Start Process");
        buttonBars.add(mp4Button);
        buttonBars.add(rgbButton);
        buttonBars.add(wavButton);
        buttonBars.add(processButton);
        add(buttonBars);

        JPanel labelPanel = new JPanel();

        JLabel mp4Label = new JLabel("No MP4");
        JLabel rgbLabel = new JLabel("No RGB");
        JLabel wavLabel = new JLabel("NO WAV");

        labelPanel.add(mp4Label);
        labelPanel.add(rgbLabel);
        labelPanel.add(wavLabel);
        add(labelPanel);


        mp4Button.addActionListener(e -> {
            JFileChooser fileChooser = new JFileChooser();

            fileChooser.setCurrentDirectory(new File(System.getProperty("user.dir")));

            FileNameExtensionFilter mp4Filter = new FileNameExtensionFilter("MP4 Files", "mp4");
            fileChooser.setFileFilter(mp4Filter);

            int returnValue = fileChooser.showOpenDialog(null);

            if (returnValue == JFileChooser.APPROVE_OPTION) {
                File selectedFile = fileChooser.getSelectedFile();
                mp4Label.setText(selectedFile.getName());
                videoUrl = selectedFile.getAbsolutePath();
            }
        });

        rgbButton.addActionListener(e -> {
            JFileChooser fileChooser = new JFileChooser();

            fileChooser.setCurrentDirectory(new File(System.getProperty("user.dir")));

            FileNameExtensionFilter mp4Filter = new FileNameExtensionFilter("RGB Files", "rgb");
            fileChooser.setFileFilter(mp4Filter);

            int returnValue = fileChooser.showOpenDialog(null);

            if (returnValue == JFileChooser.APPROVE_OPTION) {
                File selectedFile = fileChooser.getSelectedFile();
                rgbLabel.setText(selectedFile.getName());
                // TODO handle with the chosen rgb file
                rgbUrl = selectedFile.getAbsolutePath();
            }
        });

        wavButton.addActionListener(e -> {
            JFileChooser fileChooser = new JFileChooser();

            fileChooser.setCurrentDirectory(new File(System.getProperty("user.dir")));

            FileNameExtensionFilter mp4Filter = new FileNameExtensionFilter("WAV Files", "wav");
            fileChooser.setFileFilter(mp4Filter);

            int returnValue = fileChooser.showOpenDialog(null);

            if (returnValue == JFileChooser.APPROVE_OPTION) {
                File selectedFile = fileChooser.getSelectedFile();
                wavLabel.setText(selectedFile.getName());

                // TODO handle with the chosen wav file
            }
        });

        processButton.addActionListener(e -> {
            videoPlayer.init(ProcessTool.INSTANCE.processfile(rgbUrl), videoUrl);
            cardLayout.show(cardPanel, "VP");
        });
    }
}