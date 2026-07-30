using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

internal static class Launcher
{
    [STAThread]
    private static void Main()
    {
        try
        {
            string root = AppContext.BaseDirectory;
            string python = Path.Combine(root, "runtime", "pythonw.exe");
            string launcher = Path.Combine(root, "app", "launcher.py");

            var startInfo = new ProcessStartInfo
            {
                FileName = python,
                Arguments = "\"" + launcher + "\"",
                WorkingDirectory = root,
                UseShellExecute = false,
                CreateNoWindow = true
            };
            Process.Start(startInfo);
        }
        catch (Exception error)
        {
            MessageBox.Show(
                "Band Lyric Sync를 시작하지 못했습니다.\n\n" + error.Message,
                "Band Lyric Sync",
                MessageBoxButtons.OK,
                MessageBoxIcon.Error
            );
        }
    }
}
