<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <!-- Define namespaces for JATS and BioC -->
    <xsl:namespace-alias stylesheet-prefix="jats" result-prefix="http://www.ncbi.nlm.nih.gov/JATS1" />
    <xsl:namespace-alias stylesheet-prefix="bioc" result-prefix="http://bioc.sourceforge.net/schema/bioc.xsd" />

    <!-- Match the root element of the JATS XML -->
    <xsl:template match="jats:article">
        <!-- Create the root element of the BioC XML -->
        <bioc:collection>
            <!-- Process metadata if available -->
            <bioc:source> <!-- Define the source of the document -->
                <xsl:value-of select="jats:journal-meta/jats:journal-title-group/jats:journal-title" />
            </bioc:source>
            <!-- Process each article element -->
            <xsl:apply-templates select="jats:body/jats:sec"/>
        </bioc:collection>
    </xsl:template>

    <!-- Match section elements in JATS XML -->
    <xsl:template match="jats:sec">
        <!-- Create passage elements in BioC XML -->
        <bioc:document>
            <!-- Process section titles -->
            <bioc:passage> <!-- Define a passage -->
                <bioc:infon key="type">
                    <xsl:value-of select="@sec-type" /> <!-- Use section type as key -->
                </bioc:infon>
                <bioc:offset> <!-- Define the offset -->
                    <xsl:value-of select="count(preceding-sibling::jats:sec)" />
                </bioc:offset>
                <bioc:text> <!-- Define the passage text -->
                    <xsl:apply-templates />
                </bioc:text>
            </bioc:passage>
            <!-- Process subsections recursively -->
            <xsl:apply-templates select="jats:sec"/>
        </bioc:document>
    </xsl:template>

    <!-- Match generic text nodes -->
    <xsl:template match="text()">
        <xsl:value-of select="." />
    </xsl:template>
</xsl:stylesheet>
