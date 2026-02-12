/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_COMMON_QUTILEX_HXX
#define ABCDK_COMMON_QUTILEX_HXX

#include "Qt.hxx"

#ifdef HAVE_QT5

namespace abcdk
{
    namespace common
    {
        namespace QUtilEx
        {
            static inline std::vector<uint8_t> readDataFromFile(const QString &path)
            {
                QFile file(path);
                if (!file.open(QIODevice::ReadOnly))
                {
                    return {};
                }

                std::vector<uint8_t> buffer(file.size());
                if (buffer.empty())
                {
                    return buffer;
                }

                file.read((char*)buffer.data(), buffer.size());
                return buffer;
            }

            static inline QString loadStyleSheet(const QString &file)
            {
                QFile r(file);

                if (r.open(QFile::ReadOnly | QFile::Text))
                    return (QString)r.readAll();

                return QString("");
            }

            static inline QString loadStyleSheet(const std::string &file)
            {
                return loadStyleSheet((QString)file.c_str());
            }

            static inline void drawText(QImage &image, int x, int y, const QString &text,
                                        const QColor &fontColor = Qt::white,
                                        int fontPixelSize = 16,
                                        int fontWeight = QFont::Medium,
                                        const QString &fontFamily = "Arial")
            {
                QPainter painter(&image);
                painter.setRenderHint(QPainter::Antialiasing);
                painter.setRenderHint(QPainter::TextAntialiasing);
                painter.setRenderHint(QPainter::SmoothPixmapTransform);

                painter.setPen(fontColor);

                QFont font(fontFamily);
                font.setPixelSize(fontPixelSize);
                font.setWeight(fontWeight);
                painter.setFont(font);

                QFontMetrics metrics(font);
                int textWidth = metrics.horizontalAdvance(text); // 计算宽度
                int textHeight = metrics.height();               // 计算行高

                if (x < 0)
                    x = image.size().width() / 2 - textWidth / 2; // x方向居中左上角.

                if (y < 0)
                    y = image.size().height() / 2 - textHeight / 2; // y方向居中左上角.

                QRect rect(x, y, textWidth, textHeight);
                painter.drawText(rect, Qt::AlignCenter, text);
            }

            static inline QString currentDateTime(const QString &format = "hh:mm:ss.zzz")
            {
                return QDateTime::currentDateTime().toString(format);
            }

            static inline QIcon getIcon(const QString &file,int w = 256, int h = 256)
            {
                QPixmap tmp = QPixmap(file);
                if (tmp.isNull())
                    return QIcon("");

                return QIcon(tmp.scaled(QSize(w, h), Qt::KeepAspectRatio, Qt::SmoothTransformation));
            }

        } // namespace QUtilEx

    } // namespace common
} // namespace abcdk

#endif // HAVE_QT5

#endif // ABCDK_COMMON_QUTILEX_HXX