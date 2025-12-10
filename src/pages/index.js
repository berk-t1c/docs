import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import Heading from '@theme/Heading';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          Event-Driven Perception for the Tactical Edge
        </Heading>
        <p className="hero__subtitle">
          The battlefield demands sub-millisecond response in extreme environments with zero cloud dependency. 
          We deliver ultra-low power vision systems capturing 10,000 fps equivalent at &lt;2W — enabling real-time AI 
          where GPUs physically fail.
        </p>
        <div className={styles.heroStats}>
          <div className={styles.statItem}>
            <div className={styles.statValue}>&lt;2W Power</div>
          </div>
          <div className={styles.statItem}>
            <div className={styles.statValue}>~1ms Latency</div>
          </div>
          <div className={styles.statItem}>
            <div className={styles.statValue}>120dB Dynamic Range</div>
          </div>
          <div className={styles.statItem}>
            <div className={styles.statValue}>10,000 fps Equivalent</div>
          </div>
        </div>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/intro">
            Explore Documentation →
          </Link>
          <Link
            className="button button--outline button--secondary button--lg"
            href="https://www.type1compute.com/"
            style={{marginLeft: '1rem'}}>
            Visit Website →
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title} - ${siteConfig.tagline}`}
      description="Type 1 Compute delivers ultra-low power vision systems for edge AI. Brain-inspired processors enabling real-time perception at 10,000 fps equivalent with <2W power consumption.">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
        <section className={styles.performanceSection}>
          <div className="container">
            <div className="row">
              <div className="col col--12">
                <Heading as="h2" className="text--center margin-bottom--lg">
                  Performance Comparison
                </Heading>
                <p className="text--center margin-bottom--xl">
                  Gesture Recognition Benchmark (DVS128 Dataset)<br/>
                  Energy efficiency measured on neuromorphic-emulated FPGA hardware
                </p>
                <div className={styles.performanceChart}>
                  <div className={styles.performanceBar}>
                    <div className={styles.performanceLabel}>Type 1 Compute FPGA</div>
                    <div className={styles.performanceBarContainer}>
                      <div className={styles.performanceBarFill} style={{width: '100%'}}>
                        <span className={styles.performanceValue}>75.76 GOP/s/W</span>
                      </div>
                    </div>
                  </div>
                  <div className={styles.performanceBar}>
                    <div className={styles.performanceLabel}>Jetson Nano</div>
                    <div className={styles.performanceBarContainer}>
                      <div className={styles.performanceBarFill} style={{width: '10.6%'}}>
                        <span className={styles.performanceValue}>8.00 GOP/s/W</span>
                      </div>
                    </div>
                  </div>
                  <div className={styles.performanceBar}>
                    <div className={styles.performanceLabel}>RTX 3060</div>
                    <div className={styles.performanceBarContainer}>
                      <div className={styles.performanceBarFill} style={{width: '5.0%'}}>
                        <span className={styles.performanceValue}>3.81 GOP/s/W</span>
                      </div>
                    </div>
                  </div>
                  <div className={styles.performanceBar}>
                    <div className={styles.performanceLabel}>Intel i9</div>
                    <div className={styles.performanceBarContainer}>
                      <div className={styles.performanceBarFill} style={{width: '0.4%'}}>
                        <span className={styles.performanceValue}>0.31 GOP/s/W</span>
                      </div>
                    </div>
                  </div>
                </div>
                <p className="text--center margin-top--lg">
                  <strong>244× more efficient than CPU, 9.5× more efficient than Jetson</strong>
                </p>
              </div>
            </div>
          </div>
        </section>
        <section className={styles.applicationsSection}>
          <div className="container">
            <Heading as="h2" className="text--center margin-bottom--xl">
              Validated Performance Across Industries
            </Heading>
            <div className="row">
              <div className="col col--6 margin-bottom--lg">
                <div className={styles.applicationCard}>
                  <Heading as="h3">Defense Applications</Heading>
                  <ul className={styles.applicationList}>
                    <li>
                      <strong>Gesture Recognition:</strong> 75.76 GOP/s/W demonstrated<br/>
                      <span className={styles.applicationDesc}>Enable split-second human-machine interaction for safety-critical military systems without cloud dependency</span>
                    </li>
                    <li>
                      <strong>Object Detection:</strong> SpikeYOLO 5.7× efficiency improvement<br/>
                      <span className={styles.applicationDesc}>Track fast-moving threats in real-time using 5× less power than conventional AI systems</span>
                    </li>
                    <li>
                      <strong>UAV Control:</strong> 7× more efficient than Jetson Nano<br/>
                      <span className={styles.applicationDesc}>Navigate drones in GPS-denied environments at &lt;1W power consumption</span>
                    </li>
                    <li>
                      <strong>Radiation Tolerant Compute:</strong> 5x higher MTBF<br/>
                      <span className={styles.applicationDesc}>Sustain deterministic, low-latency inference under high-radiation LEO/HEO spacecraft environments</span>
                    </li>
                  </ul>
                </div>
              </div>
              <div className="col col--6 margin-bottom--lg">
                <div className={styles.applicationCard}>
                  <Heading as="h3">Research Applications</Heading>
                  <ul className={styles.applicationList}>
                    <li>
                      <strong>HVAC Optimization:</strong> 21% energy reduction<br/>
                      <span className={styles.applicationDesc}>Cut building energy costs while improving occupant comfort through real-time predictive control</span>
                    </li>
                    <li>
                      <strong>Medical EEG:</strong> 90% Parkinson's detection accuracy<br/>
                      <span className={styles.applicationDesc}>Enable bedside neurological diagnostics without cloud processing, preserving patient privacy</span>
                    </li>
                    <li>
                      <strong>Surgical Tracking:</strong> &lt;3ms latency instrument detection<br/>
                      <span className={styles.applicationDesc}>Provide surgeons with instant safety alerts and workflow analytics during procedures</span>
                    </li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>
        <section className={styles.advantageSection}>
          <div className="container">
            <div className="row">
              <div className="col col--12">
                <Heading as="h2" className="text--center margin-bottom--lg">
                  The Event-Driven Advantage
                </Heading>
                <p className="text--center margin-bottom--xl" style={{fontSize: '1.1rem'}}>
                  Traditional cameras capture full frames 30 times per second—wasting power processing unchanged pixels. 
                  Event-driven sensors fire only when pixels detect change, enabling:
                </p>
                <div className={styles.advantageExample}>
                  <div className="row">
                    <div className="col col--6">
                      <div className={styles.comparisonCard}>
                        <Heading as="h3">Frame-Based (Wasteful)</Heading>
                        <p>Processes all pixels every frame</p>
                      </div>
                    </div>
                    <div className="col col--6">
                      <div className={styles.comparisonCard}>
                        <Heading as="h3">Event-Driven (Efficient)</Heading>
                        <p>Processes only changed pixels</p>
                      </div>
                    </div>
                  </div>
                </div>
                <p className="text--center margin-top--xl" style={{fontSize: '1.1rem'}}>
                  <strong>SPACE SURVEILLANCE:</strong> Track satellites moving 17,000 mph against bright Earth or dark space 
                  simultaneously—impossible for frame cameras that saturate or lose contrast.
                </p>
                <p className="text--center margin-top--md">
                  This same principle scales to autonomous vehicles, industrial systems, drones, and robotics requiring instant obstacle detection.
                </p>
              </div>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}
