import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Word Counter Tool',
  description: 'A simple word counter tool with history tracking',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen">{children}</body>
    </html>
  );
}
